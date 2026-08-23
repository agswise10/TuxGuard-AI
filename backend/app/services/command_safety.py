"""
Command Safety Analyzer.

This is the deterministic, rule-based safety gate for Safe Command
Execution. It is intentionally independent of the LLM: the model that
generates or explains a command is never trusted to also judge its own
safety. Every command - whether AI-generated or typed directly by the
user - passes through `analyze_command()` here before it can ever be
executed, and again immediately before execution in case it was edited
in between.

Two layers of rules:
  1. BLOCK_RULES - patterns that are always refused, no matter what.
     Confirming does not override these; there is no way to execute a
     blocked command through this API.
  2. WARNING_RULES - patterns that are risky but sometimes legitimate
     (e.g. `sudo`, `curl | bash`, `chmod -R`). These raise the risk level
     and add a specific warning, but can still be executed if the user
     explicitly confirms.

Risk levels, low to high: safe < low < medium < high < blocked.
"""

import re
from dataclasses import dataclass, field

RiskLevel = str  # "safe" | "low" | "medium" | "high" | "blocked"

_RISK_ORDER = {"safe": 0, "low": 1, "medium": 2, "high": 3, "blocked": 4}


@dataclass
class Rule:
    name: str
    pattern: "re.Pattern[str]"
    description: str
    severity: RiskLevel  # "low" | "medium" | "high" for warnings, "blocked" for block rules


@dataclass
class MatchedRule:
    rule: str
    description: str
    severity: RiskLevel


@dataclass
class AnalysisResult:
    command: str
    blocked: bool
    risk_level: RiskLevel
    matched_rules: list[MatchedRule] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "blocked": self.blocked,
            "risk_level": self.risk_level,
            "matched_rules": [
                {"rule": m.rule, "description": m.description, "severity": m.severity}
                for m in self.matched_rules
            ],
            "warnings": self.warnings,
        }


def _re(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.IGNORECASE)


# Matches a "root-like" target as a whole token: exactly `/`, `/*`, `~`, or
# `$HOME` with nothing else appended to it (so `/home/user` or `~/Downloads`
# do NOT match, but a bare `/`, `/*`, `~`, or `$HOME` does). Used only via
# `_targets_root()` below - never as a raw substring search.
_ROOT_TOKEN_RE = _re(r"(?:^|\s)(/|/\*|~|\$HOME)(?=\s|;|&|\||`|$)")
_RECURSIVE_FLAG_RE = _re(r"(?:^|\s)-[a-z]*[rR][a-z]*(?:\s|$)|--recursive\b")
_FORCE_FLAG_RE = _re(r"(?:^|\s)-[a-z]*f[a-z]*(?:\s|$)|--force\b")


def _split_shell_segments(command: str) -> list[str]:
    """Best-effort split of a command line into individual sub-commands on
    shell separators (;, &&, ||, &, |), so each is analyzed on its own and
    a root-like target in one sub-command can't be "hidden" behind an
    earlier harmless one."""
    return re.split(r"&&|\|\||[;&|]", command)


def _rm_targets_root(segment: str) -> bool:
    """True if `segment` is an `rm` invocation that is both recursive+forced
    AND targets a whole root-like path ('/', '/*', '~', or '$HOME') as a
    standalone argument - not merely containing '/' somewhere in a longer,
    perfectly normal path like /home/user/tempfolder."""
    match = re.search(r"\brm\b(.*)", segment, re.IGNORECASE)
    if not match:
        return False
    args = match.group(1)
    if not (_RECURSIVE_FLAG_RE.search(args) and _FORCE_FLAG_RE.search(args)):
        return False
    return bool(_ROOT_TOKEN_RE.search(args))


def _mv_or_chmod_targets_root(segment: str, command_word: str, require_recursive: bool) -> bool:
    """Generic helper for mv/chmod/chown-style rules that should only fire
    when a root-like path is the actual, standalone target - not a
    substring of a normal deep path."""
    match = re.search(rf"\b{command_word}\b(.*)", segment, re.IGNORECASE)
    if not match:
        return False
    args = match.group(1)
    if require_recursive and not _RECURSIVE_FLAG_RE.search(args):
        return False
    return bool(_ROOT_TOKEN_RE.search(args))


# ---------------------------------------------------------------------------
# Hard-block rules: matching ANY of these means the command can NEVER be
# executed through this API, confirmation or not.
# ---------------------------------------------------------------------------
BLOCK_RULES: list[Rule] = [
    Rule(
        "fork_bomb",
        _re(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&?\s*\}\s*;\s*:"),
        "Classic fork bomb pattern - rapidly exhausts system resources and can crash the machine.",
        "blocked",
    ),
    Rule(
        "no_preserve_root",
        _re(r"--no-preserve-root"),
        "Explicitly disables rm's built-in protection against deleting '/' - never legitimate outside kernel/rootfs build scripts.",
        "blocked",
    ),
    Rule(
        "mkfs_any",
        _re(r"\bmkfs(\.\w+)?\b"),
        "Formats a filesystem, permanently destroying all data on the target partition or device.",
        "blocked",
    ),
    Rule(
        "dd_to_block_device",
        _re(r"\bdd\b.*\bof=\s*/dev/(sd[a-z]|nvme\d+n\d+|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d+)(?!\d*p\d)\b"),
        "Writes raw data directly to an entire block device with `dd`, which will destroy its partition table and all data.",
        "blocked",
    ),
    Rule(
        "redirect_to_block_device",
        _re(r"[>]{1,2}\s*/dev/(sd[a-z]|nvme\d+n\d+|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d+)\b"),
        "Redirects output directly onto a whole block device, which will corrupt or destroy its contents.",
        "blocked",
    ),
    Rule(
        "wipefs",
        _re(r"\bwipefs\b"),
        "Erases filesystem/partition signatures, making data on the device inaccessible.",
        "blocked",
    ),
    Rule(
        "partition_table_write",
        _re(r"\b(sgdisk\s+--zap|parted\s+.*\b(mklabel|rm)\b|fdisk\s+.*\b(--wipe|--wipe-partitions)\b)"),
        "Rewrites or wipes a disk's partition table, which can make all data on it unreachable.",
        "blocked",
    ),
    Rule(
        "shutdown_reboot",
        _re(r"\b(shutdown|reboot|halt|poweroff)\b|\bsystemctl\s+(poweroff|halt|reboot)\b|\binit\s+[06]\b"),
        "Shuts down or restarts the machine, which would interrupt this session and any running work.",
        "blocked",
    ),
]

# Root-targeting rules for rm / chmod / chown / mv are implemented as
# functions (see `_rm_targets_root`, `_mv_or_chmod_targets_root` above)
# rather than single regexes, so a bare "/" or "~" is only matched as a
# whole, standalone argument - never as a substring of an ordinary deep
# path like "/home/user/tempfolder". They are evaluated per shell segment
# inside `analyze_command()`.
_ROOT_TARGET_CHECKS = [
    ("rm_rf_root", lambda seg: _rm_targets_root(seg),
     "Recursive, forced deletion targeting the root filesystem, a wildcard root, or the home directory - would destroy the system or all user data."),
    ("chmod_chown_root_recursive", lambda seg: (
        _mv_or_chmod_targets_root(seg, "chmod", require_recursive=True)
        or _mv_or_chmod_targets_root(seg, "chown", require_recursive=True)
    ), "Recursively changes permissions/ownership of the entire root filesystem, which will break the system."),
    ("mv_root_to_devnull", lambda seg: (
        _mv_or_chmod_targets_root(seg, "mv", require_recursive=False) and "/dev/null" in seg
    ), "Moves the entire root filesystem into /dev/null, effectively deleting everything."),
]

# ---------------------------------------------------------------------------
# Warning rules: risky but sometimes legitimate. Raise the risk level and
# explain why, but do not block execution outright.
# ---------------------------------------------------------------------------
WARNING_RULES: list[Rule] = [
    Rule(
        "sudo_usage",
        _re(r"\bsudo\b"),
        "Runs with elevated (root) privileges - mistakes here can affect the whole system.",
        "medium",
    ),
    Rule(
        "recursive_force_flags",
        _re(r"(-[a-z]*r[a-z]*f[a-z]*\b|-[a-z]*f[a-z]*r[a-z]*\b|--recursive\b|--force\b)"),
        "Uses recursive and/or force flags, which skip normal safety confirmations and can affect many files at once.",
        "medium",
    ),
    Rule(
        "pipe_remote_script_to_shell",
        _re(r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(bash|sh|zsh)\b"),
        "Downloads a script from the internet and pipes it directly into a shell - you cannot review what it does before it runs.",
        "high",
    ),
    Rule(
        "dd_generic",
        _re(r"\bdd\b"),
        "`dd` performs low-level, unbuffered data copying - a wrong `of=` target can silently destroy data.",
        "high",
    ),
    Rule(
        "device_path_reference",
        _re(r"/dev/(sd[a-z]|nvme\d+n\d+|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d+)\b"),
        "References a raw block device path directly - double-check this is the intended disk.",
        "high",
    ),
    Rule(
        "kill_signal_broad",
        _re(r"\b(kill\s+-9\s+-1|killall\s+-9\b|pkill\s+-9\s+-f\s+\.\*)"),
        "Force-kills a broad set of processes, which can crash unrelated services or the desktop session.",
        "medium",
    ),
    Rule(
        "firewall_disable",
        _re(r"\b(ufw\s+disable|iptables\s+-F|iptables\s+--flush)\b"),
        "Disables or flushes firewall rules, which can expose the machine on the network.",
        "medium",
    ),
    Rule(
        "package_purge",
        _re(r"\bapt(-get)?\s+(purge|remove)\s+.*(-y|--yes)\b.*\*"),
        "Purges packages matching a wildcard, which can remove more than intended.",
        "medium",
    ),
    Rule(
        "crontab_remove_all",
        _re(r"\bcrontab\s+-r\b"),
        "Deletes ALL of the current user's scheduled cron jobs at once, with no confirmation and no backup.",
        "medium",
    ),
    Rule(
        "overwrite_redirect",
        _re(r"[^>]>[^>]"),
        "Uses a single '>' redirect, which overwrites (truncates) the target file if it already exists.",
        "low",
    ),
]


def list_rules() -> dict:
    """Return every rule name/description/severity, for a transparency
    endpoint - so users (and hackathon judges) can see exactly what this
    system blocks and warns on, without digging through source code."""
    return {
        "block_rules": [
            {"name": r.name, "description": r.description, "severity": r.severity}
            for r in BLOCK_RULES
        ]
        + [
            {"name": name, "description": description, "severity": "blocked"}
            for name, _check, description in _ROOT_TARGET_CHECKS
        ],
        "warning_rules": [
            {"name": r.name, "description": r.description, "severity": r.severity}
            for r in WARNING_RULES
        ],
    }


def _highest(levels: list[RiskLevel]) -> RiskLevel:
    if not levels:
        return "safe"
    return max(levels, key=lambda lvl: _RISK_ORDER.get(lvl, 0))


def analyze_command(command: str) -> AnalysisResult:
    """Run a command string through every safety rule and return a verdict.

    Never raises. An empty/whitespace-only command is treated as blocked
    (there is nothing safe to execute).
    """
    command = (command or "").strip()
    if not command:
        return AnalysisResult(
            command=command,
            blocked=True,
            risk_level="blocked",
            matched_rules=[
                MatchedRule("empty_command", "No command was provided.", "blocked")
            ],
            warnings=["No command was provided."],
        )

    matched: list[MatchedRule] = []
    seen_rule_names: set[str] = set()

    for rule in BLOCK_RULES:
        if rule.pattern.search(command):
            matched.append(MatchedRule(rule.name, rule.description, rule.severity))
            seen_rule_names.add(rule.name)

    for segment in _split_shell_segments(command):
        for name, check, description in _ROOT_TARGET_CHECKS:
            if name in seen_rule_names:
                continue
            if check(segment):
                matched.append(MatchedRule(name, description, "blocked"))
                seen_rule_names.add(name)

    if matched:
        # Any block-rule match short-circuits: this command can never run.
        return AnalysisResult(
            command=command,
            blocked=True,
            risk_level="blocked",
            matched_rules=matched,
            warnings=[m.description for m in matched],
        )

    warning_matches: list[MatchedRule] = []
    for rule in WARNING_RULES:
        if rule.pattern.search(command):
            warning_matches.append(MatchedRule(rule.name, rule.description, rule.severity))

    risk_level = _highest([m.severity for m in warning_matches]) if warning_matches else "safe"

    return AnalysisResult(
        command=command,
        blocked=False,
        risk_level=risk_level,
        matched_rules=warning_matches,
        warnings=[m.description for m in warning_matches],
    )

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import Monitor, Remove, Restore


class MyBlueHeuristicAgent(BaseAgent):
    CRITICAL_HOST = "Op_Server0"
    RESTORE_THRESHOLD = 3
    COOLDOWN_TURNS = 2
    PROACTIVE_SWEEP_INTERVAL = 2
    PROACTIVE_SWEEP_STEPS = 12
    PROACTIVE_PATH = [
        "Enterprise0",
        "Enterprise1",
        "Enterprise2",
        "Op_Host0",
        "Op_Host1",
        "Op_Host2",
    ]

    def __init__(self):
        self.suspicious_hosts = []
        self.queued_hosts = set()
        self.host_hits = {}
        self.host_cooldowns = {}
        self.step_count = 0
        self.proactive_index = 0
        self.last_action = None
        self.last_target = None

    def train(self, results: Results):
        pass

    def _host_priority(self, hostname: str):
        if hostname == self.CRITICAL_HOST:
            return 0
        if hostname.startswith("Op_Host"):
            return 1
        if hostname.startswith("Enterprise"):
            return 2
        if hostname == "Defender":
            return 3
        if hostname.startswith("User"):
            return 4
        return 5

    def _extract_session(self, action_space):
        sessions = action_space.get("session", {}) if isinstance(action_space, dict) else {}
        if isinstance(sessions, dict) and sessions:
            return list(sessions.keys())[0]
        return 0

    def _update_suspicious_hosts(self, observation):
        for obs_key, host_info in observation.items():
            if obs_key == "success" or not isinstance(host_info, dict):
                continue

            hostname = host_info.get("System info", {}).get("Hostname", obs_key)
            if not hostname or hostname == "User0":
                continue

            processes = host_info.get("Processes", [])
            has_suspicious = any(isinstance(proc, dict) and "PID" in proc for proc in processes)
            if not has_suspicious:
                continue

            self.host_hits[hostname] = self.host_hits.get(hostname, 0) + 1
            if hostname not in self.queued_hosts:
                self.suspicious_hosts.append(hostname)
                self.queued_hosts.add(hostname)

    def _tick_cooldowns(self):
        expired = [h for h, t in self.host_cooldowns.items() if t <= 1]
        for host in expired:
            del self.host_cooldowns[host]
        for host in self.host_cooldowns:
            self.host_cooldowns[host] -= 1

    def _pick_target_host(self):
        if not self.suspicious_hosts:
            return None

        self.suspicious_hosts.sort(
            key=lambda h: (self._host_priority(h), -self.host_hits.get(h, 0), h)
        )
        for idx, host in enumerate(self.suspicious_hosts):
            if host == self.CRITICAL_HOST or not self.host_cooldowns.get(host, 0):
                self.suspicious_hosts.pop(idx)
                self.queued_hosts.discard(host)
                return host
        return None

    def _pick_proactive_host(self):
        host = self.PROACTIVE_PATH[self.proactive_index]
        self.proactive_index = (self.proactive_index + 1) % len(self.PROACTIVE_PATH)
        return host

    def _choose_action(self, target_host, session):
        hit_count = self.host_hits.get(target_host, 0)
        priority = self._host_priority(target_host)

        self.host_cooldowns[target_host] = self.COOLDOWN_TURNS
        self.last_target = target_host

        if target_host == self.CRITICAL_HOST:
            self.last_action = "Restore"
            return Restore(hostname=target_host, agent="Blue", session=session)

        if priority <= 2 and hit_count >= self.RESTORE_THRESHOLD:
            self.last_action = "Restore"
            return Restore(hostname=target_host, agent="Blue", session=session)

        self.last_action = "Remove"
        return Remove(hostname=target_host, agent="Blue", session=session)

    def get_action(self, observation, action_space):
        self.step_count += 1
        self._tick_cooldowns()

        # The environment auto-runs Monitor for Blue every step and merges
        # the result into the observation, so we always have fresh data.
        self._update_suspicious_hosts(observation)

        session = self._extract_session(action_space)

        # Always handle Op_Server0 immediately — every turn it's impacted costs -10.
        if self.CRITICAL_HOST in self.queued_hosts:
            self.suspicious_hosts = [
                h for h in self.suspicious_hosts if h != self.CRITICAL_HOST
            ]
            self.queued_hosts.discard(self.CRITICAL_HOST)
            return self._choose_action(self.CRITICAL_HOST, session)

        if self.suspicious_hosts:
            target_host = self._pick_target_host()
            if target_host is not None:
                return self._choose_action(target_host, session)

        # Proactive sweep for predictable Red path hosts in early game.
        if (
            self.step_count <= self.PROACTIVE_SWEEP_STEPS
            and self.step_count % self.PROACTIVE_SWEEP_INTERVAL == 0
        ):
            proactive_host = self._pick_proactive_host()
            self.last_action = "Remove"
            self.last_target = proactive_host
            return Remove(hostname=proactive_host, agent="Blue", session=session)

        self.last_action = "Monitor"
        return Monitor(agent="Blue", session=session)

    def end_episode(self):
        self.suspicious_hosts = []
        self.queued_hosts = set()
        self.host_hits = {}
        self.host_cooldowns = {}
        self.step_count = 0
        self.proactive_index = 0
        self.last_action = None
        self.last_target = None

    def set_initial_values(self, action_space, observation):
        pass

"""从训练配置/checkpoint 加载 OpenPI 策略并启动 WebSocket 服务。

Galaxea 不需要在这里增加一个新的 server 实现：只要通过
``--policy.config=pi05_galaxea_r1_multitask`` 选择本地新增的训练配置，
``create_trained_policy`` 就会自动装配 Galaxea transforms、stats 校验和
模型，然后由通用 ``WebsocketPolicyServer`` 对外提供推理。
"""

import dataclasses
import enum
import ipaddress
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import galaxea_r1_deployment
from openpi.serving import galaxea_r1_pro_deployment
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """官方默认策略对应的环境类型。"""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """指定训练配置名和 checkpoint 目录。"""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """使用官方预设环境的默认 checkpoint。"""


@dataclasses.dataclass
class Args:
    """服务端启动参数。Galaxea 通常使用 ``policy:checkpoint`` 分支。"""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Listen address. Loopback is the safe cloud default; put a TLS reverse
    # proxy in front, or explicitly select 0.0.0.0 when using a VPN/firewall.
    host: str = "127.0.0.1"
    # Read the shared API key from this environment variable. The secret is
    # intentionally not accepted as a command-line value, so it does not leak
    # through shell history or process listings.
    api_key_env: str | None = "OPENPI_API_KEY"
    # Explicit escape hatch for trusted local/LAN experiments. A non-loopback
    # listener without an API key otherwise fails closed.
    allow_unauthenticated: bool = False
    # Limit a single observation message. Three 224x224 RGB images are well
    # below this value; the margin allows protocol metadata and future cameras.
    max_request_size_mib: float = 32.0
    # Run one synthetic R1 inference before opening the port. This removes the
    # first-request JAX compilation delay from the robot control session.
    warmup: bool = False
    warmup_prompt: str = "pick up the object"
    # Full tracebacks contain local paths and implementation details. Keep this
    # false for every Internet-facing deployment.
    expose_errors: bool = False
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """按官方环境枚举加载默认策略。"""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据命令行的 Default/Checkpoint 变体构造策略。"""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def _is_loopback_host(host: str) -> bool:
    """返回监听地址是否只允许本机连接。"""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _load_api_key(args: Args) -> str | None:
    """从环境变量读取 API key，并对非本机无鉴权监听执行 fail-closed。"""
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if api_key == "":
        raise ValueError(f"{args.api_key_env} is set but empty")
    if api_key is None and not _is_loopback_host(args.host) and not args.allow_unauthenticated:
        env_hint = args.api_key_env or "OPENPI_API_KEY"
        raise ValueError(
            f"Refusing unauthenticated non-loopback listener {args.host!r}. "
            f"Set {env_hint}, bind to 127.0.0.1 behind a TLS proxy, or explicitly pass "
            "--allow-unauthenticated on a trusted network."
        )
    if api_key is None:
        logging.warning("Policy server authentication is disabled; listener=%s", args.host)
    return api_key


def main(args: Args) -> None:
    """加载策略、复制 metadata 并启动长期运行的 WebSocket 服务。"""
    if args.max_request_size_mib <= 0:
        raise ValueError("max_request_size_mib must be positive")
    api_key = _load_api_key(args)

    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.warmup:
        if policy_metadata.get("robot") == "r1_pro":
            galaxea_r1_pro_deployment.warmup_policy(policy, prompt=args.warmup_prompt)
        else:
            galaxea_r1_deployment.warmup_policy(policy, prompt=args.warmup_prompt)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Creating server (hostname=%s, local_ip=%s, listen=%s:%d, auth=%s)",
        hostname,
        local_ip,
        args.host,
        args.port,
        "api-key" if api_key else "disabled",
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        api_key=api_key,
        max_request_size=int(args.max_request_size_mib * 1024 * 1024),
        expose_errors=args.expose_errors,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

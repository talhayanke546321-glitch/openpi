# Galaxea R1 π0.5 云端推理服务

本目录只覆盖 GPU 云服务器。机器人侧 ROS 2 bridge、动作安全器和任务状态机
需要在 R1/控制电脑上单独部署。

## 安全边界

- 策略进程默认只监听 `127.0.0.1:8000`。
- 推理连接要求 `Authorization: Api-Key <secret>`；`/healthz` 不要求鉴权。
- API key 只从环境变量读取，不出现在 shell history 或进程参数中。
- 公网访问使用 `wss://`。API key 在明文 `ws://` 中不能防止链路窃听。
- 单条观测默认限制为 32 MiB，服务端异常只写日志，不向客户端返回 traceback。
- 当前 checkpoint metadata 明确标记 `training_domain=simulation` 和
  `real_robot_validated=false`；服务就绪不代表策略已经通过真机验证。

## 1. 本机启动与验证

```bash
cd /home/vipuser/robotics/openpi
export OPENPI_API_KEY='使用密码管理器生成的长随机值'

uv run scripts/serve_galaxea_r1_cloud.py
```

该入口会检查 checkpoint 的 params、stats 和 stats metadata，加载模型，执行
一次 `3×224×224 RGB + 14维状态 -> 15×14动作` 的合成预热，通过后才监听端口。

另开终端执行：

```bash
cd /home/vipuser/robotics/openpi
export OPENPI_API_KEY='与服务端相同的值'
uv run scripts/smoke_test_galaxea_r1_server.py
curl http://127.0.0.1:8000/healthz
```

## 2. 公网接入方式

推荐域名 + Caddy：

1. 将 `deploy/Caddyfile.example` 复制到 Caddy 配置并替换域名；
2. 云安全组只开放 443，8000 不对公网开放；
3. R1 客户端连接 `wss://实际域名` 并携带 API key。

如果通过 Tailscale/WireGuard，服务可显式监听 `0.0.0.0`，但安全组仍应只允许
VPN 网段：

```bash
uv run scripts/serve_galaxea_r1_cloud.py --host 0.0.0.0
```

不要把未加密的 8000 端口直接暴露给整个互联网。

## 3. systemd

```bash
sudo install -d -m 700 /etc/openpi
sudo install -m 600 deploy/galaxea-r1.env.example /etc/openpi/galaxea-r1.env
# 编辑 /etc/openpi/galaxea-r1.env，写入真实 OPENPI_API_KEY

sudo install -m 644 deploy/galaxea-r1-openpi.service.example \
  /etc/systemd/system/galaxea-r1-openpi.service
sudo systemctl daemon-reload
sudo systemctl enable --now galaxea-r1-openpi.service
sudo systemctl status galaxea-r1-openpi.service
journalctl -u galaxea-r1-openpi.service -f
```

服务启动允许最多 900 秒，用于首次 checkpoint 恢复和 JAX 编译。

## 4. R1 客户端必须验证的 metadata

连接后服务端会公布：协议版本、机器人型号、控制器、关节顺序、单位、三路
图像键、动作维度和 action horizon。真机客户端应 fail-closed：缺字段、顺序
不同或 checkpoint 标记不兼容时，不允许发布任何 ROS 2 动作。

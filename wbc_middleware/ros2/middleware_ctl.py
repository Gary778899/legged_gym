from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_srvs.srv import Trigger

from wbc_middleware.core.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_STARTUP_TRIGGER_SERVICE,
    DEFAULT_STATUS_TRIGGER_SERVICE,
    DEFAULT_STOP_TRIGGER_SERVICE,
)
from wbc_middleware.core.metadata_loader import load_yaml_config


class MiddlewareCtlNode(Node):
    def __init__(self) -> None:
        super().__init__('middleware_ctl')

    def call_trigger(
        self,
        service_name: str,
        *,
        wait_timeout_s: float,
        call_timeout_s: float,
    ) -> Trigger.Response:
        client = self.create_client(Trigger, service_name)
        try:
            if not client.wait_for_service(timeout_sec=wait_timeout_s):
                raise RuntimeError(
                    f"Service '{service_name}' not available within {wait_timeout_s:.1f}s."
                )
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=call_timeout_s)
            if not future.done():
                future.cancel()
                raise RuntimeError(
                    f"Service '{service_name}' did not respond within {call_timeout_s:.1f}s."
                )
            if future.exception() is not None:
                raise RuntimeError(
                    f"Service '{service_name}' failed: {future.exception()}"
                )
            response = future.result()
            if response is None:
                raise RuntimeError(f"Service '{service_name}' returned no response.")
            return response
        finally:
            self.destroy_client(client)


def _load_configured_services(config_path: str) -> dict[str, str]:
    config = load_yaml_config(config_path)
    return {
        'start': str(config.get('startup_trigger_service', DEFAULT_STARTUP_TRIGGER_SERVICE)),
        'stop': str(config.get('stop_trigger_service', DEFAULT_STOP_TRIGGER_SERVICE)),
        'status': str(config.get('status_trigger_service', DEFAULT_STATUS_TRIGGER_SERVICE)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='middleware_ctl',
        description='Operator CLI for the WBC middleware control node.',
    )
    parser.add_argument(
        'command',
        choices=('start', 'stop', 'status'),
        help='Operation to perform.',
    )
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_PATH,
        help=f'Path to middleware YAML config. Default: {DEFAULT_CONFIG_PATH}',
    )
    parser.add_argument(
        '--wait-timeout',
        type=float,
        default=3.0,
        help='Seconds to wait for the target service to appear.',
    )
    parser.add_argument(
        '--call-timeout',
        type=float,
        default=3.0,
        help='Seconds to wait for the service reply after the call is sent.',
    )
    return parser


def _print_action_result(command: str, response: Trigger.Response) -> int:
    label = 'START' if command == 'start' else 'STOP'
    if response.success:
        print(f'[{label}] OK: {response.message}')
        return 0
    print(f'[{label}] REJECTED: {response.message}', file=sys.stderr)
    return 2


def _print_status(response: Trigger.Response) -> int:
    if not response.success:
        print(f'[STATUS] ERROR: {response.message}', file=sys.stderr)
        return 2
    try:
        payload = json.loads(response.message)
    except json.JSONDecodeError:
        print('[STATUS] ERROR: middleware returned an unreadable status payload.', file=sys.stderr)
        return 2

    startup_state = payload.get('startup_state', 'UNKNOWN')
    safety_state = payload.get('safety_state', 'unknown')
    has_fresh_state = _bool_text(payload.get('has_fresh_state'))
    imu_ready = _bool_text(payload.get('imu_ready'))
    joints_ready = _bool_text(payload.get('joints_ready'))
    can_start = _bool_text(payload.get('can_start'))
    can_stop = _bool_text(payload.get('can_stop'))
    imu_age = _format_optional_seconds(payload.get('imu_age_s'))
    joint_age = _format_optional_seconds(payload.get('joint_age_s'))
    timeout_s = _format_optional_seconds(payload.get('state_timeout_s'))
    pose_error = _format_optional_float(payload.get('max_default_pose_error'))
    interactive = _bool_text(payload.get('interactive_command_active'))
    stop_reason = payload.get('last_safe_stop_reason') or '-'
    safe_hold_mode = payload.get('safe_hold_mode', '-')

    print(f'State: {startup_state}')
    print(f'Safety: {safety_state}')
    print(
        f'Readiness: fresh_state={has_fresh_state}, imu_ready={imu_ready}, '
        f'joints_ready={joints_ready}'
    )
    print(
        f'State age: imu={imu_age}, joints={joint_age}, timeout={timeout_s}'
    )
    print(
        f'Control: can_start={can_start}, can_stop={can_stop}, '
        f'interactive_command={interactive}'
    )
    print(f'Safe hold: mode={safe_hold_mode}, last_reason={stop_reason}')
    print(f'Default pose error: {pose_error}')
    return 0


def _bool_text(value: Any) -> str:
    return 'yes' if bool(value) else 'no'


def _format_optional_seconds(value: Any) -> str:
    if value is None:
        return '-'
    try:
        return f'{float(value):.3f}s'
    except (TypeError, ValueError):
        return str(value)


def _format_optional_float(value: Any) -> str:
    if value is None:
        return '-'
    try:
        return f'{float(value):.4f} rad'
    except (TypeError, ValueError):
        return str(value)


def main(args: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(args=args)
    services = _load_configured_services(parsed.config)

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = MiddlewareCtlNode()
    try:
        response = node.call_trigger(
            services[parsed.command],
            wait_timeout_s=parsed.wait_timeout,
            call_timeout_s=parsed.call_timeout,
        )
        if parsed.command == 'status':
            return _print_status(response)
        return _print_action_result(parsed.command, response)
    except RuntimeError as exc:
        print(f'[{parsed.command.upper()}] ERROR: {exc}', file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)


if __name__ == '__main__':
    raise SystemExit(main())

"""Small chat-completions HTTP client; secrets stay in memory, never in artifacts."""
import json
import os
import ssl
import sys
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


def load_settings(path):
    """Read simple KEY=value dotenv syntax; process environment takes precedence."""
    values = {}
    if Path(path).is_file():
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            key, separator, value = line.partition('=')
            if not separator:
                raise ValueError('Invalid .env assignment (contents withheld)')
            key, value = key.strip(), value.strip()
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError('Unclosed .env quote (contents withheld)')
                value = value[1:-1]
            else:
                value = value.split(' #', 1)[0].rstrip()
            values[key] = value
    result = {}
    for key in ('RUBRIC_API_KEY', 'RUBRIC_API_BASE_URL', 'RUBRIC_API_MODEL'):
        result[key] = os.environ.get(key) or values.get(key, '')
        if not result[key].strip():
            raise ValueError('Missing local setting: ' + key)
    return result


def endpoint(base):
    try:
        parts = urlsplit(base)
        _ = parts.port
    except ValueError:
        raise ValueError('Invalid API URL (contents withheld)') from None
    if (parts.scheme not in ('http', 'https') or not parts.hostname or
            parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('API URL must be http(s), without credentials, query or fragment')
    return base.rstrip('/') if parts.path.rstrip('/').endswith('/chat/completions') else base.rstrip('/') + '/chat/completions'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward an Authorization header to a redirected host.


class ChatAPI:
    def __init__(self, settings, config, opener_factory=None):
        self.key = settings['RUBRIC_API_KEY']
        self.url = endpoint(settings['RUBRIC_API_BASE_URL'])
        self.model = settings['RUBRIC_API_MODEL']
        if self.key in self.url or self.key in self.model:
            raise ValueError('Credentials must not appear in the API URL or model identifier')
        self.config = config
        self.opener_factory = opener_factory or self.verified_opener

    def verified_opener(self):
        backend = self.config.get('api_tls_backend', 'urllib_default')
        if backend == 'urllib_default':
            return build_opener(NoRedirect())
        if backend != 'windows_truststore' or sys.platform != 'win32':
            raise ValueError('Unsupported verified TLS backend on this platform')
        try:
            import truststore
        except ImportError:
            # The bundled Python has pip's native Windows certificate verifier.
            from pip._vendor import truststore
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ValueError('Native TLS must verify certificates and hostnames')
        return build_opener(NoRedirect(), HTTPSHandler(context=context))

    def describe(self):
        return {'protocol': 'chat_completions', 'endpoint': self.url, 'requested_model': self.model,
                'max_tokens': self.config.get('api_max_tokens', 2048),
                'temperature': self.config.get('api_temperature'),
                'json_object_requested': self.config.get('api_json_mode', False),
                'timeout_seconds': self.config.get('api_timeout_seconds', 120),
                'max_attempts': self.config.get('api_max_attempts', 3),
                'tls_backend': self.config.get('api_tls_backend', 'urllib_default'),
                'thinking_requested': self.config.get('api_thinking'),
                'certificate_verification': True}

    def __call__(self, messages):
        body = {'model': self.model, 'messages': messages,
                'max_tokens': self.config.get('api_max_tokens', 2048)}
        if 'api_temperature' in self.config:
            body['temperature'] = self.config['api_temperature']
        if self.config.get('api_json_mode', False):
            body['response_format'] = {'type': 'json_object'}
        if 'api_thinking' in self.config:
            if self.config['api_thinking'] not in ('enabled', 'disabled'):
                raise ValueError('api_thinking must be enabled or disabled')
            body['thinking'] = {'type': self.config['api_thinking']}
        attempts = self.config.get('api_max_attempts', 3)
        if not isinstance(attempts, int) or not 1 <= attempts <= 5:
            raise ValueError('api_max_attempts must be 1..5')
        for attempt in range(attempts):
            request = Request(self.url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                              headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}, method='POST')
            try:
                with self.opener_factory().open(request, timeout=self.config.get('api_timeout_seconds', 120)) as response:
                    raw = response.read().decode('utf-8')
                # Do not persist a server response that echoes the credential.
                if self.key in raw:
                    raise RuntimeError('API response echoed a credential; response withheld')
                value = json.loads(raw)
                choice = value['choices'][0]
                content = choice['message']['content']
                # Save metadata/usage even when the output was truncated or empty.
                # The dataset processor rejects it after durable response storage.
                return {'content': content if isinstance(content, str) else '', 'returned_model': value.get('model'),
                        'response_id': value.get('id'), 'usage': value.get('usage'),
                        'finish_reason': choice.get('finish_reason')}
            except HTTPError as exc:
                retry = exc.code == 429 or 500 <= exc.code < 600
                if not retry or attempt + 1 == attempts:
                    raise RuntimeError(f'API HTTP {exc.code}; error body and credentials withheld') from None
            except (URLError, TimeoutError, OSError):
                if attempt + 1 == attempts:
                    raise RuntimeError('API network/timeout failure; connection details withheld') from None
            except (ValueError, KeyError, IndexError, TypeError):
                raise RuntimeError('API returned invalid/incomplete chat-completions output; details withheld') from None
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError('API request failed')

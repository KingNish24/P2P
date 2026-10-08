export async function onRequest(context) {
  const { request } = context;
  const url = new URL(request.url);

  // Preflight OPTIONS handler
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, HEAD',
        'Access-Control-Allow-Headers': '*',
        'Access-Control-Max-Age': '86400',
      },
    });
  }

  // Proxy Hugging Face requests
  const hfPath = url.pathname.replace(/^\/hf\//, '');
  const targetUrl = `https://huggingface.co/${hfPath}${url.search}`;

  const forwardHeaders = new Headers();
  forwardHeaders.set('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36');
  if (request.headers.has('range')) {
    forwardHeaders.set('range', request.headers.get('range'));
  }
  if (request.headers.has('accept')) {
    forwardHeaders.set('accept', request.headers.get('accept'));
  }
  if (request.headers.has('accept-encoding')) {
    forwardHeaders.set('accept-encoding', request.headers.get('accept-encoding'));
  }

  const response = await fetch(targetUrl, {
    method: request.method,
    headers: forwardHeaders,
    redirect: 'follow',
  });

  const responseHeaders = new Headers(response.headers);
  responseHeaders.set('Access-Control-Allow-Origin', '*');
  responseHeaders.set('Cross-Origin-Resource-Policy', 'cross-origin');

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: responseHeaders,
  });
}

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

  const forwardHeaders = new Headers(request.headers);
  forwardHeaders.delete('host');

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

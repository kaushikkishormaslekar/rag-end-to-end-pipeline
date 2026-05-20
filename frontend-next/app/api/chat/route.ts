const gatewayUrl = process.env.SPRING_GATEWAY_URL ?? "http://127.0.0.1:8080";

export async function POST(request: Request) {
  const authorization = request.headers.get("authorization");
  const headers = new Headers({
    accept: "application/json",
    "content-type": "application/json",
  });

  if (authorization) {
    headers.set("authorization", authorization);
  }

  const response = await fetch(`${gatewayUrl}/api/chat`, {
    method: "POST",
    headers,
    body: await request.text(),
    cache: "no-store",
  });

  return new Response(await response.text(), {
    status: response.status,
    headers: {
      "content-type": response.headers.get("content-type") ?? "application/json",
    },
  });
}

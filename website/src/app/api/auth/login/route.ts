import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { email, password } = body;

  if (!email || !password) {
    return NextResponse.json(
      { message: "Email and password are required" },
      { status: 400 }
    );
  }

  // TODO: Validate credentials against PostgreSQL via the Flask backend
  // For now, accept any well-formed request as a placeholder
  return NextResponse.json(
    { message: "Login successful", user: { email } },
    { status: 200 }
  );
}

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { name, email, password } = body;

  if (!name || !email || !password) {
    return NextResponse.json(
      { message: "Name, email, and password are required" },
      { status: 400 }
    );
  }

  if (password.length < 8) {
    return NextResponse.json(
      { message: "Password must be at least 8 characters" },
      { status: 400 }
    );
  }

  // TODO: Create user in PostgreSQL via Flask backend
  // For now, accept any well-formed request as a placeholder
  return NextResponse.json(
    { message: "Account created", user: { name, email } },
    { status: 201 }
  );
}

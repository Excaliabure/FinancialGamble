from axiomtradeapi import AxiomTradeClient
import asyncio





client = AxiomTradeClient(
    auth_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoZW50aWNhdGVkVXNlcklkIjoiYjUxYzU0MDYtNzE0ZS00ZDI4LTgxNzktNjRiZmY2YWI2YmJjIiwiaWF0IjoxNzU1MzA5NDkzLCJleHAiOjE3NTUzMTA0NTN9.e297DSHyQvnkAnTGXDrjMepTsYVQtLEMuZDZ1QHg9VA",
    
    
    
    
    refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyZWZyZXNoVG9rZW5JZCI6ImY5NTZjY2M1LThkYTktNGZmMS1iNGVlLTc5YmZhMTUwNjE2OCIsImlhdCI6MTc1NTMwODE2Nn0.I5nYI4MghC9iKjyOptyKEnHTguVRcpMyfAYnO6tK_UI"
    )

async def handle_tokens(tokens):
    print(f"New tokens: {len(tokens)}")

await client.subscribe_new_tokens(handle_tokens)



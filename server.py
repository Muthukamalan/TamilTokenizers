from minbpe import RegexTokenizer
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import uvicorn


app = FastAPI()

# Initialize the tokenizer
tokenizer = RegexTokenizer()
tokenizer.load("./models/regex.model")


class TokenInfo(BaseModel):
    token_id: int
    token_bytes: str
    token_text: str


class TokenRequest(BaseModel):
    text: str


class TokenResponse(BaseModel):
    token_ids: List[int]
    token_details: List[TokenInfo]
    full_text: str


@app.post("/encode")
def encode_text(request: TokenRequest) -> TokenResponse:
    # Get token IDs
    token_ids = tokenizer.encode(request.text)

    # Get detailed information for each token
    token_details = []
    for token_id in token_ids:
        # Get the bytes for this token
        token_bytes = tokenizer.vocab[token_id]

        # Convert bytes to string representation, replacing invalid chars
        token_text = token_bytes.decode("utf-8")

        token_details.append(
            TokenInfo(
                token_id=token_id,
                token_bytes=str(token_bytes),  # Convert bytes to string representation
                token_text=token_text,
            )
        )

    # Reconstruct the full text from tokens for verification
    decoded_text = tokenizer.decode(token_ids)

    return TokenResponse(
        token_ids=token_ids, token_details=token_details, full_text=decoded_text
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

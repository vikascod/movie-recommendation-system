from pydantic import BaseModel

class MovieName(BaseModel):
    movie_name: str

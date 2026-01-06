from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Portfolio Chatbot Agent")

# CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    thread_id: str = "default_thread"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Agentic Chatbot is running."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        initial_state = {"messages": [HumanMessage(content=request.query)]}
        # In a real app complexity, you'd handle thread_id for memory in LangGraph checkpointer
        # For this simple v1, we just invoke the graph statelessly or with in-memory state if we configured checkpointer.
        # The graph as defined in agent.py doesn't use checkpointer yet, so it is stateless conversation turn.
        
        result = graph.invoke(initial_state)
        final_message = result["messages"][-1]
        
        return {"response": final_message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

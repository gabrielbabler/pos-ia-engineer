import { AIMessage, SystemMessage } from "langchain";
import { type GraphState } from "../graph.ts";
import { stat } from "fs";

export function fallbackNode(state: GraphState): GraphState {
    const message = "Unknown command. Try 'make this uppercase' or 'convert to lowercase'";
    const fallbackMessage = new AIMessage(message).content.toString()

    return {
        ...state, 
        output: fallbackMessage,
        messages: [
            ...state.messages,
        ]
    }
}
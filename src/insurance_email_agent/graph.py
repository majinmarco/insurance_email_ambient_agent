"""
Insurance Email Ambient Agent
1. Receives email json
2. Extracts data from email body; classifies email based on body
3. Classifies each received document; extracts data from each document
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START, StateGraph
from pydantic import BaseModel

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

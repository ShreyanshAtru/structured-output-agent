"""Typed tool interfaces, execution, and synthetic demo tool."""
from .base import Tool, ToolCall, ToolDefinition, ToolResult
from .executor import ToolExecutor
from .ticket_context import TicketContextInput, TicketContextOutput, TicketContextTool

__all__ = ["Tool", "ToolCall", "ToolDefinition", "ToolResult", "ToolExecutor",
           "TicketContextInput", "TicketContextOutput", "TicketContextTool"]

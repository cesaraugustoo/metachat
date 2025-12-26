from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime
import uuid
from .base import Agent

class IterativeAgent(Agent):
    """
    Agent that solves problems through an iterative monologue, 
    supporting tool use via special tags.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add logging configuration
        self.log_dir = Path(kwargs.get('log_dir', "experiments/logs/cot_iterative")) / self.model.model_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug = kwargs.get('debug', False)
        self.max_iterations = kwargs.get('max_iterations', 20)
        
        if not self.system_prompt:
            self.system_prompt = """You are an expert in optics and photonics engaging in a continuous conversation to help users with their optics and photonics questions. 
You have access to neural network-based design APIs. You can iteratively talk to yourself and have an internal monologue.

Guidelines:
0. Think step by step. Break down complex problems into steps and plan your approach before solving.
1. If you need to design a metalens or superpixel, use these neural network tools:
   - For metalenses, use: <tool>neural_design
   design_metalens(refractive_index, lens_diameter [m], focal_length [m], thickness [m], operating_wavelength [m])
   </tool>
   - For superpixels, use: <tool>neural_design
   design_superpixel(refractive_index, length [m], incident_angle [deg], diffraction_angle [deg], thickness [m], operating_wavelength [m])
   </tool>
2. Return the final answer wrapped in <response> tags. Make sure your code prints the final answer in the correct units
3. If no calculations are needed, simply state the answer directly
4. You can only use ONE type of tag per message
5. Make sure to convert intermediate results to the correct units before using them to prevent multiplication or function unit mismatch errors

IMPORTANT: Any text not wrapped in tags will be treated as your internal thoughts and planning. Only text within <response> tags will be shown to the user.
"""

    async def _log_conversation(self, problem_id: str, messages: List[Dict[str, str]]):
        """Log conversation to a JSON file."""
        log_file = self.log_dir / f"{problem_id}.json"
        
        existing_log = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_log = json.load(f)
            except json.JSONDecodeError:
                existing_log = []
                
        existing_log.extend(messages)
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_log, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            if self.debug:
                print(f"Error writing to log file: {e}")

    async def solve(self, problem: str, problem_id: Optional[str] = None, temperature: float = 1.0, disable_cache: bool = False) -> Dict[str, Any]:
        if not problem_id:
            problem_id = str(uuid.uuid4())
            
        await self._log_conversation(problem_id, [{
            "role": "user",
            "content": problem,
            "timestamp": datetime.now().isoformat()
        }])

        if disable_cache:
            timestamp = datetime.now().isoformat()
            problem = f"Date submitted: {timestamp}\n\n{problem}"

        messages = self._format_messages(problem)
        iteration_count = 0
        conversation = []

        while True:
            iteration_count += 1
            
            if iteration_count > self.max_iterations:
                await self._log_conversation(problem_id, [{
                    "role": "error",
                    "content": "Maximum iterations reached",
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])

                return {
                    "status": "error",
                    "error": "Maximum iterations reached",
                    "conversation": conversation,
                    "problem_id": problem_id
                }
            
            if self.debug:
                print(f"\n=== Iteration {iteration_count} ===")
            
            response_content = await self._call_model(messages, temperature=temperature)

            await self._log_conversation(problem_id, [{
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration_count
            }])

            if "<response>" in response_content:
                response = response_content.split("<response>")[1].split("</response>")[0].strip()
                
                await self._log_conversation(problem_id, [{
                    "role": "response",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                return {
                    "solution": response,
                    "metadata": {
                        "method": "iterative",
                        "num_iterations": iteration_count,
                        "conversation": conversation,
                        "problem_id": problem_id
                    },
                    "tool_calls": self.tool_calls
                }

            # If no tags present, treat as thinking/planning
            if not any(tag in response_content for tag in ["<tool>", "<response>"]):
                await self._log_conversation(problem_id, [{
                    "role": "thinking",
                    "content": response_content,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": iteration_count
                }])
                
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": "Continue with your approach. If you need to reply to the user with an answer or need clarification, respond with <response> tags."})
                continue

            # Process tool calls
            tool_found = False
            for tool_name, tool in self.tools.items():
                if f"<tool>{tool_name}" in response_content:
                    tool_found = True
                    start = response_content.find(f"<tool>{tool_name}")
                    end = response_content.find("</tool>", start)
                    if end != -1:
                        code_block = response_content[start + len(f"<tool>{tool_name}"):end].strip()
                        
                        await self._log_conversation(problem_id, [{
                            "role": "tool_call",
                            "tool": tool_name,
                            "code": code_block,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        # Execute tool
                        result = await tool.execute(code_block)
                        
                        await self._log_conversation(problem_id, [{
                            "role": "tool_response",
                            "tool": tool_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                            "iteration": iteration_count
                        }])
                        
                        messages.append({"role": "assistant", "content": response_content})
                        messages.append({"role": "user", "content": f"Tool output: {result}"})
            
            if not tool_found and "<tool>" in response_content:
                # Handle unknown tool
                messages.append({"role": "assistant", "content": response_content})
                messages.append({"role": "user", "content": "Error: Unknown tool tag or malformed tool call."})

            conversation.append({
                "iteration": iteration_count,
                "content": response_content,
                "timestamp": datetime.now().isoformat()
            })

"""
Persistent Task Completion Agent Framework

A robust, production-ready autonomous agent that can accomplish any given task
through iterative interaction with an LLM, continuing until successful completion.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime


class TaskStatus(Enum):
    """Enumeration for task completion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationResult(Enum):
    """Enumeration for validation results."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    UNCLEAR = "unclear"


@dataclass
class TaskAttempt:
    """Represents a single attempt at completing a task."""
    attempt_number: int
    prompt_sent: str
    llm_response: str
    validation_result: ValidationResult
    validation_reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the autonomous agent."""
    max_attempts: int = 10
    validation_threshold: float = 0.8
    retry_delay: float = 1.0
    enable_learning: bool = True
    log_level: str = "INFO"
    timeout_seconds: int = 300
    

class LLMInterface(Protocol):
    """Protocol defining the interface for LLM communication."""
    
    def send_request(self, prompt: str, context: Optional[str] = None) -> str:
        """Send a request to the LLM and return the response."""
        ...
    
    def is_available(self) -> bool:
        """Check if the LLM is available for requests."""
        ...


class MockLLM:
    """Mock LLM implementation for testing purposes."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or []
        self.call_count = 0
        
    def send_request(self, prompt: str, context: Optional[str] = None) -> str:
        """Return a mock response."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = f"Mock response {self.call_count + 1} for prompt: {prompt[:50]}..."
        
        self.call_count += 1
        return response
    
    def is_available(self) -> bool:
        return True


class ValidationEngine:
    """Handles task completion validation."""
    
    VALIDATION_PROMPT_TEMPLATE = """
You are a task completion validator. Your job is to determine if a given task has been successfully completed based on the LLM's response.

ORIGINAL TASK:
{original_task}

LLM'S RESPONSE:
{llm_response}

PREVIOUS ATTEMPTS (if any):
{previous_attempts}

Please evaluate whether the task has been completed successfully and respond with:

1. STATUS: One of [SUCCESS, PARTIAL, FAILURE, UNCLEAR]
2. REASONING: Brief explanation of your assessment
3. SUGGESTIONS: If not successful, provide specific suggestions for improvement

Format your response as:
STATUS: [your_status]
REASONING: [your_reasoning]
SUGGESTIONS: [your_suggestions_if_applicable]
"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.logger = logging.getLogger(__name__ + ".ValidationEngine")
    
    def validate_completion(
        self, 
        original_task: str, 
        llm_response: str, 
        previous_attempts: List[TaskAttempt]
    ) -> Tuple[ValidationResult, str]:
        """Validate if the task has been completed successfully."""
        
        try:
            # Format previous attempts for context
            attempts_context = self._format_previous_attempts(previous_attempts)
            
            validation_prompt = self.VALIDATION_PROMPT_TEMPLATE.format(
                original_task=original_task,
                llm_response=llm_response,
                previous_attempts=attempts_context
            )
            
            validation_response = self.llm.send_request(validation_prompt)
            return self._parse_validation_response(validation_response)
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult.UNCLEAR, f"Validation failed due to error: {e}"
    
    def _format_previous_attempts(self, attempts: List[TaskAttempt]) -> str:
        """Format previous attempts for validation context."""
        if not attempts:
            return "No previous attempts."
        
        formatted = []
        for attempt in attempts[-3:]:  # Only show last 3 attempts
            formatted.append(
                f"Attempt {attempt.attempt_number}: "
                f"{attempt.validation_result.value} - {attempt.validation_reasoning[:100]}..."
            )
        
        return "\n".join(formatted)
    
    def _parse_validation_response(self, response: str) -> Tuple[ValidationResult, str]:
        """Parse the validation response to extract status and reasoning."""
        try:
            lines = response.strip().split('\n')
            status_line = next((line for line in lines if line.startswith('STATUS:')), '')
            reasoning_line = next((line for line in lines if line.startswith('REASONING:')), '')
            
            # Extract status
            status_text = status_line.replace('STATUS:', '').strip().upper()
            validation_result = ValidationResult.SUCCESS
            
            if 'PARTIAL' in status_text:
                validation_result = ValidationResult.PARTIAL
            elif 'FAILURE' in status_text or 'FAILED' in status_text:
                validation_result = ValidationResult.FAILURE
            elif 'UNCLEAR' in status_text:
                validation_result = ValidationResult.UNCLEAR
            
            # Extract reasoning
            reasoning = reasoning_line.replace('REASONING:', '').strip()
            
            return validation_result, reasoning or response[:200]
            
        except Exception as e:
            self.logger.warning(f"Failed to parse validation response: {e}")
            return ValidationResult.UNCLEAR, f"Parse error: {response[:200]}"


class TaskManager:
    """Manages task state and history."""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__ + ".TaskManager")
    
    def create_task(self, task_description: str, task_id: Optional[str] = None) -> str:
        """Create a new task and return its ID."""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        self.tasks[task_id] = {
            'id': task_id,
            'description': task_description,
            'status': TaskStatus.PENDING,
            'attempts': [],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        self.logger.info(f"Created task {task_id}: {task_description[:50]}...")
        return task_id
    
    def update_task_status(self, task_id: str, status: TaskStatus):
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = status
            self.tasks[task_id]['updated_at'] = datetime.now()
    
    def add_attempt(self, task_id: str, attempt: TaskAttempt):
        """Add an attempt to a task's history."""
        if task_id in self.tasks:
            self.tasks[task_id]['attempts'].append(attempt)
            self.tasks[task_id]['updated_at'] = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task information by ID."""
        return self.tasks.get(task_id)
    
    def get_attempts(self, task_id: str) -> List[TaskAttempt]:
        """Get all attempts for a task."""
        task = self.get_task(task_id)
        return task['attempts'] if task else []


class PersistentTaskAgent:
    """
    Main autonomous agent class that persistently works on tasks until completion.
    """
    
    def __init__(self, llm_interface: LLMInterface, config: Optional[AgentConfig] = None):
        self.llm = llm_interface
        self.config = config or AgentConfig()
        self.task_manager = TaskManager()
        self.validator = ValidationEngine(llm_interface)
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__ + ".PersistentTaskAgent")
        
        self.logger.info("Initialized PersistentTaskAgent")
    
    def execute_task(self, task_description: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to execute a task persistently until completion.
        
        Args:
            task_description: The task to be completed
            task_id: Optional task ID for tracking
            
        Returns:
            Dictionary containing task results and metadata
        """
        # Create or get task
        if task_id is None:
            task_id = self.task_manager.create_task(task_description)
        
        self.task_manager.update_task_status(task_id, TaskStatus.IN_PROGRESS)
        
        start_time = time.time()
        
        try:
            result = self._execute_task_loop(task_id, task_description)
            execution_time = time.time() - start_time
            
            return {
                'task_id': task_id,
                'status': result['status'],
                'final_response': result.get('final_response'),
                'attempts_made': result['attempts_made'],
                'execution_time': execution_time,
                'success': result['status'] == TaskStatus.COMPLETED
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED)
            
            return {
                'task_id': task_id,
                'status': TaskStatus.FAILED,
                'error': str(e),
                'attempts_made': len(self.task_manager.get_attempts(task_id)),
                'execution_time': time.time() - start_time,
                'success': False
            }
    
    def _execute_task_loop(self, task_id: str, task_description: str) -> Dict[str, Any]:
        """Execute the main task completion loop."""
        attempts = []
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            self.logger.info(f"Starting attempt {attempt_num}/{self.config.max_attempts}")
            
            # Check LLM availability
            if not self.llm.is_available():
                self.logger.warning("LLM not available, waiting...")
                time.sleep(self.config.retry_delay)
                continue
            
            # Generate prompt based on previous attempts
            prompt = self._generate_prompt(task_description, attempts)
            
            # Get LLM response
            start_time = time.time()
            try:
                llm_response = self.llm.send_request(prompt)
                execution_time = time.time() - start_time
            except Exception as e:
                self.logger.error(f"LLM request failed: {e}")
                time.sleep(self.config.retry_delay)
                continue
            
            # Validate the response
            validation_result, validation_reasoning = self.validator.validate_completion(
                task_description, llm_response, attempts
            )
            
            # Create attempt record
            attempt = TaskAttempt(
                attempt_number=attempt_num,
                prompt_sent=prompt,
                llm_response=llm_response,
                validation_result=validation_result,
                validation_reasoning=validation_reasoning,
                execution_time=execution_time
            )
            
            attempts.append(attempt)
            self.task_manager.add_attempt(task_id, attempt)
            
            self.logger.info(f"Attempt {attempt_num} result: {validation_result.value}")
            
            # Check if task is completed
            if validation_result == ValidationResult.SUCCESS:
                self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
                return {
                    'status': TaskStatus.COMPLETED,
                    'final_response': llm_response,
                    'attempts_made': attempt_num
                }
            
            # Add delay between attempts
            if attempt_num < self.config.max_attempts:
                time.sleep(self.config.retry_delay)
        
        # Max attempts reached without success
        self.task_manager.update_task_status(task_id, TaskStatus.FAILED)
        return {
            'status': TaskStatus.FAILED,
            'attempts_made': self.config.max_attempts,
            'final_response': attempts[-1].llm_response if attempts else None
        }
    
    def _generate_prompt(self, task_description: str, previous_attempts: List[TaskAttempt]) -> str:
        """Generate an improved prompt based on previous attempts."""
        base_prompt = f"Task: {task_description}\n\nPlease complete this task."
        
        if not previous_attempts:
            return base_prompt
        
        # Add context from previous attempts
        context = "\n\nPrevious attempts and feedback:\n"
        for attempt in previous_attempts[-3:]:  # Only use last 3 attempts
            context += f"""
Attempt {attempt.attempt_number}:
Response: {attempt.llm_response[:200]}...
Validation: {attempt.validation_result.value} - {attempt.validation_reasoning}
"""
        
        improvement_note = """
Based on the previous attempts and validation feedback, please provide an improved response that addresses the identified issues.
"""
        
        return base_prompt + context + improvement_note
    
    def get_task_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of task execution."""
        task = self.task_manager.get_task(task_id)
        if not task:
            return None
        
        attempts = task['attempts']
        return {
            'task_id': task_id,
            'description': task['description'],
            'status': task['status'].value,
            'total_attempts': len(attempts),
            'created_at': task['created_at'],
            'updated_at': task['updated_at'],
            'success_rate': sum(1 for a in attempts if a.validation_result == ValidationResult.SUCCESS) / len(attempts) if attempts else 0,
            'last_attempt': attempts[-1] if attempts else None
        }


# Example usage and test cases
def main():
    """Example usage of the PersistentTaskAgent."""
    
    # Create a mock LLM with realistic responses
    mock_responses = [
        "I'll help you write a Python function. Here's a basic function that adds two numbers: def add(a, b): return a + b",
        "Let me improve that. Here's a more robust function with type hints and documentation:\n\ndef add_numbers(a: float, b: float) -> float:\n    \"\"\"Add two numbers and return the result.\"\"\"\n    return a + b",
        "Here's a comprehensive solution with error handling:\n\ndef add_numbers(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:\n    \"\"\"\n    Add two numbers with type checking and error handling.\n    \n    Args:\n        a: First number\n        b: Second number\n    \n    Returns:\n        Sum of a and b\n    \n    Raises:\n        TypeError: If inputs are not numbers\n    \"\"\"\n    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):\n        raise TypeError('Both arguments must be numbers')\n    return a + b"
    ]
    
    llm = MockLLM(mock_responses)
    
    # Create agent with custom config
    config = AgentConfig(
        max_attempts=5,
        retry_delay=0.1,  # Faster for testing
        log_level="INFO"
    )
    
    agent = PersistentTaskAgent(llm, config)
    
    # Test cases
    test_tasks = [
        "Write a Python function that adds two numbers with proper error handling and documentation",
        "Create a simple calculator class with basic arithmetic operations",
        "Explain the concept of recursion with a practical example"
    ]
    
    results = []
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'='*60}")
        print(f"EXECUTING TASK {i}: {task}")
        print(f"{'='*60}")
        
        result = agent.execute_task(task)
        results.append(result)
        
        print(f"\nTask Result:")
        print(f"Status: {result['status'].value if hasattr(result['status'], 'value') else result['status']}")
        print(f"Success: {result['success']}")
        print(f"Attempts: {result['attempts_made']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        
        if result['success']:
            print(f"Final Response: {result['final_response'][:200]}...")
        
        # Get task summary
        summary = agent.get_task_summary(result['task_id'])
        if summary:
            print(f"Success Rate: {summary['success_rate']:.2%}")
    
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful_tasks = sum(1 for r in results if r['success'])
    total_attempts = sum(r['attempts_made'] for r in results)
    
    print(f"Total Tasks: {len(results)}")
    print(f"Successful: {successful_tasks}")
    print(f"Success Rate: {successful_tasks/len(results):.2%}")
    print(f"Total Attempts: {total_attempts}")
    print(f"Average Attempts per Task: {total_attempts/len(results):.1f}")


if __name__ == "__main__":
    main()

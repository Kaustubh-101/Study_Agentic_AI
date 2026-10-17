from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
try:
    from phi.model.google import Gemini
except Exception:
    Gemini = None

try:
    from phi.tools.duckduckgo import DuckDuckGo
    DDG_AVAILABLE = True
except Exception:
    DuckDuckGo = None
    DDG_AVAILABLE = False


import yaml
import os

class StudyAgents:
    def __init__(self, topic, subject_category, knowledge_level, learning_goal, 
                 time_available, learning_style, model_name="gpt-4o", provider="openai"):
        """
        Initialize study assistant agents with student information and model configuration.
        
        Args:
            topic (str): The specific topic or subject to study
            subject_category (str): The category of the subject
            knowledge_level (str): Student's current knowledge level
            learning_goal (str): What the student wants to achieve
            time_available (str): How much time the student has
            learning_style (str): Student's preferred learning style
            model_name (str): The model to use
            provider (str): The AI provider ("openai" or "groq")
        """
        self.topic = topic
        self.subject_category = subject_category
        self.knowledge_level = knowledge_level
        self.learning_goal = learning_goal
        self.time_available = time_available
        self.learning_style = learning_style
        self.model_name = model_name
        self.provider = provider
        self.personas = self._load_personas()
    
    def _load_personas(self):
        """
        Load personas from the YAML file.
        
        Returns:
            dict: A dictionary of personas with system prompts
        """
        config_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            return config.get("personas", {})
    
    def _get_learning_style_info(self):
        """
        Load learning style-specific information from the YAML file.
        
        Returns:
            dict: Learning style configuration
        """
        config_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
            return config.get("learning_styles", {}).get(self.learning_style, {})
    
    def _get_model(self, temperature=0.7):
        """
        Get the appropriate model based on the provider.
        
        Args:
            temperature (float): The temperature setting for the model
            
        Returns:
            Model: The configured model instance
        """
        if self.provider == "gemini":
            try:
                from phi.model.google import Gemini
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                return Gemini(id=self.model_name or "gemini-2.0-flash", temperature=temperature, api_key=api_key)
            except Exception as e:
                raise ImportError(
                    "To use Gemini, the `google-generativeai` package is required. "
                    "Please install it by running: `pip install google-generativeai`"
                ) from e
        elif self.provider == "groq":
            return Groq(id=self.model_name or "qwen/qwen3.8-27b", temperature=temperature, max_tokens=900)
        elif self.provider == "openai":
            return OpenAIChat(id=self.model_name or "gpt-4o", temperature=temperature)
        else:
            return Groq(id=self.model_name or "qwen/qwen3.8-27b", temperature=temperature, max_tokens=900)


    
    def student_analyzer_agent(self):
        """
        Create a student analyzer agent that assesses learning needs and gaps.
        
        Returns:
            Agent: A student analysis-focused agent
        """
        system_prompt = self.personas.get("student_analyzer", {}).get("system_prompt", "")
        learning_style_info = self._get_learning_style_info()
        
        full_prompt = f"""{system_prompt}
        
        You are analyzing a student who wants to learn about {self.topic}.
        
        STUDENT PROFILE:
        - Current Knowledge Level: {self.knowledge_level}
        - Learning Goal: {self.learning_goal}
        - Available Time: {self.time_available}
        - Learning Style: {self.learning_style}
        - Learning Style Notes: {learning_style_info.get('description', '')}
        """
        
        return Agent(
            model=self._get_model(temperature=0.6),
            system_prompt=full_prompt
        )
    
    def roadmap_creator_agent(self):
        """
        Create a roadmap creator agent that designs personalized learning paths.
        
        Returns:
            Agent: A roadmap creation-focused agent
        """
        system_prompt = self.personas.get("roadmap_creator", {}).get("system_prompt", "")
        learning_style_info = self._get_learning_style_info()
        recommendations = learning_style_info.get('recommendations', [])
        
        full_prompt = f"""{system_prompt}
        
        You are creating a personalized learning roadmap for {self.topic}.
        
        STUDENT CONTEXT:
        - Knowledge Level: {self.knowledge_level}
        - Learning Goal: {self.learning_goal}
        - Time Available: {self.time_available}
        - Learning Style: {self.learning_style}
        
        LEARNING STYLE RECOMMENDATIONS:
        {chr(10).join(f'- {rec}' for rec in recommendations)}
        """
        
        return Agent(
            model=self._get_model(temperature=0.7),
            system_prompt=full_prompt
        )
    
    def quiz_generator_agent(self):
        """
        Create a quiz generator agent that creates assessments and practice questions.
        
        Returns:
            Agent: A quiz generation-focused agent
        """
        system_prompt = self.personas.get("quiz_generator", {}).get("system_prompt", "")
        
        full_prompt = f"""{system_prompt}
        
        You are creating quizzes for a student learning {self.topic}.
        
        STUDENT LEVEL: {self.knowledge_level}
        
        Ensure questions are appropriate for this knowledge level and help the student 
        progress toward their goal: {self.learning_goal}
        
        IMPORTANT: You MUST respond with ONLY valid JSON. No markdown formatting, no code fences, no extra text before or after the JSON.
        """
        
        return Agent(
            model=self._get_model(temperature=0.5),
            system_prompt=full_prompt
        )
    
    def answer_evaluator_agent(self):
        """
        Create an answer evaluator agent that provides personalized feedback on quiz results.
        
        Returns:
            Agent: An answer evaluation-focused agent
        """
        system_prompt = self.personas.get("quiz_generator", {}).get("system_prompt", "")
        
        full_prompt = f"""{system_prompt}
        
        You are evaluating quiz performance for a student learning {self.topic}.
        
        STUDENT LEVEL: {self.knowledge_level}
        LEARNING GOAL: {self.learning_goal}
        
        Provide encouraging, constructive feedback. Identify patterns in mistakes
        and give actionable study recommendations.
        """
        
        return Agent(
            model=self._get_model(temperature=0.7),
            system_prompt=full_prompt
        )
    
    def tutor_agent(self):
        """
        Create a tutor agent that explains concepts and answers questions.
        
        Returns:
            Agent: A tutoring-focused agent
        """
        system_prompt = self.personas.get("tutor_agent", {}).get("system_prompt", "")
        learning_style_info = self._get_learning_style_info()
        
        full_prompt = f"""{system_prompt}
        
        You are tutoring a student on {self.topic}.
        
        STUDENT CONTEXT:
        - Knowledge Level: {self.knowledge_level}
        - Learning Style: {self.learning_style} - {learning_style_info.get('description', '')}
        
        Adapt your explanations to match their learning style and knowledge level.
        """
        
        return Agent(
            model=self._get_model(temperature=0.7),
            system_prompt=full_prompt
        )
    
    def resource_finder_agent(self):
        """
        Create a resource finder agent that searches for learning materials.
        
        Returns:
            Agent: A resource finding-focused agent with search capabilities
        """
        system_prompt = self.personas.get("resource_finder", {}).get("system_prompt", "")
        learning_style_info = self._get_learning_style_info()
        
        full_prompt = f"""{system_prompt}
        
        You are finding learning resources for {self.topic}.
        
        STUDENT PREFERENCES:
        - Knowledge Level: {self.knowledge_level}
        - Learning Style: {self.learning_style}
        - Learning Goal: {self.learning_goal}
        
        Prioritize resources that match the {self.learning_style} learning style.
        """
        
        tools = []
        if DDG_AVAILABLE and DuckDuckGo is not None:
            try:
                tools.append(DuckDuckGo())
            except Exception:
                pass

        agent_kwargs = {
            "model": self._get_model(temperature=0.6),
            "system_prompt": full_prompt
        }
        if tools:
            agent_kwargs["tools"] = tools
            agent_kwargs["show_tool_calls"] = True

        return Agent(**agent_kwargs)
    
    def rag_tutor_agent(self, knowledge_base=None):
        """
        Create a RAG-enabled tutor agent that can answer questions using uploaded documents.
        
        Args:
            knowledge_base: The knowledge base/vector store to use for RAG
            
        Returns:
            Agent: A RAG-enabled tutoring agent
        """
        system_prompt = self.personas.get("tutor_agent", {}).get("system_prompt", "")
        
        full_prompt = f"""{system_prompt}
        
        You are tutoring a student on {self.topic} using provided study materials.
        
        IMPORTANT:
        - Base your answers on the provided context from the student's documents
        - If information isn't in the provided context, acknowledge this
        - Cite specific sections when referencing the materials
        - Help the student understand the material deeply
        
        STUDENT LEVEL: {self.knowledge_level}
        """
        
        agent_config = {
            "model": self._get_model(temperature=0.6),
            "system_prompt": full_prompt
        }
        
        # Add knowledge base if provided
        if knowledge_base:
            agent_config["knowledge_base"] = knowledge_base
            agent_config["search_knowledge"] = True
        
        return Agent(**agent_config)

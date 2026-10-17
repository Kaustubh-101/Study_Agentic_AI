import yaml
import json
import re
import os
import streamlit as st
from study_agents import StudyAgents
from rag_helper import RAGHelper
from typing import Optional, Dict, Any

class StudyAssistantHandler:
    def __init__(self, topic, subject_category, knowledge_level, learning_goal, 
                 time_available, learning_style, model_name="gpt-4o", provider="openai"):
        """
        Initialize the study assistant handler.
        
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
        self.agents = StudyAgents(
            topic, subject_category, knowledge_level, learning_goal,
            time_available, learning_style, model_name, provider
        )
        self.config = self._load_config()
        self.rag_helper = None
    
    def _load_config(self):
        """
        Load the complete configuration from the YAML file.
        
        Returns:
            dict: A dictionary of configuration data
        """
        config_path = os.path.join(os.path.dirname(__file__), "prompts.yaml")
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    
    def _format_prompt(self, prompt_template, **kwargs):
        """
        Format a prompt template with variables.
        
        Args:
            prompt_template (str): The prompt template to format
            **kwargs: The variables to insert into the template
            
        Returns:
            str: The formatted prompt
        """
        return prompt_template.format(**kwargs)
    
    def _run_agent(self, agent_getter_name: str, prompt: str):
        """
        Run an agent with automatic fallback between providers if rate limits or errors occur.
        Supports: Groq → Gemini and Gemini → Groq fallback.
        """
        try:
            agent = getattr(self.agents, agent_getter_name)()
            resp = agent.run(prompt, stream=False)
            return resp.content
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit, quota exceeded, 429, or resource exhausted
            is_limit_error = any(kw in error_str for kw in ["429", "quota", "exhausted", "rate", "limit", "resource_exhausted"])
            if is_limit_error:
                # Groq hit limit → fallback to Gemini
                if self.provider == "groq":
                    gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
                    if gemini_key and not gemini_key.startswith("your_"):
                        st.warning("⚠️ **Groq rate limit reached.** Seamlessly switching to **Gemini** to continue...")
                        self.provider = "gemini"
                        self.model_name = "gemini-3.8-flash"
                        self.agents = StudyAgents(
                            self.topic, self.subject_category, self.knowledge_level, self.learning_goal,
                            self.time_available, self.learning_style, self.model_name, self.provider
                        )
                        fallback_agent = getattr(self.agents, agent_getter_name)()
                        fallback_resp = fallback_agent.run(prompt, stream=False)
                        return fallback_resp.content
                # Gemini hit limit → fallback to Groq
                elif self.provider == "gemini":
                    groq_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
                    if groq_key and not groq_key.startswith("your_"):
                        st.warning("⚠️ **Gemini limit exceeded.** Seamlessly switching to **Groq (`qwen/qwen3.8-27b`)** to continue...")
                        self.provider = "groq"
                        self.model_name = "qwen/qwen3.8-27b"
                        self.agents = StudyAgents(
                            self.topic, self.subject_category, self.knowledge_level, self.learning_goal,
                            self.time_available, self.learning_style, self.model_name, self.provider
                        )
                        fallback_agent = getattr(self.agents, agent_getter_name)()
                        fallback_resp = fallback_agent.run(prompt, stream=False)
                        return fallback_resp.content
            raise e

    def analyze_student(self):
        """
        Analyze the student's learning needs and create a profile.
        
        Returns:
            dict: Analysis results
        """
        results = {}
        
        with st.status("Analyzing your learning needs...", expanded=True) as status:
            status.update(label="Creating student profile...", state="running")
            
            analysis_prompt = self._format_prompt(
                self.config["prompts"]["student_analysis"]["base"],
                topic=self.topic,
                subject_category=self.subject_category,
                knowledge_level=self.knowledge_level,
                learning_goal=self.learning_goal,
                time_available=self.time_available,
                learning_style=self.learning_style
            )
            
            analysis_result = self._run_agent("student_analyzer_agent", analysis_prompt)
            results["analysis"] = analysis_result
            st.session_state.student_analysis = analysis_result
            
            status.update(label="Analysis complete!", state="complete")
        
        return results
    
    def create_roadmap(self, student_analysis: str):
        """
        Create a personalized learning roadmap based on student analysis.
        
        Args:
            student_analysis (str): The student analysis from analyze_student()
            
        Returns:
            dict: Roadmap results
        """
        results = {}
        
        with st.status("Creating your personalized learning roadmap...", expanded=True) as status:
            status.update(label="Designing learning path...", state="running")
            
            roadmap_prompt = self._format_prompt(
                self.config["prompts"]["roadmap_creation"]["base"],
                student_analysis=student_analysis,
                topic=self.topic,
                learning_goal=self.learning_goal,
                time_available=self.time_available,
                knowledge_level=self.knowledge_level
            )
            
            roadmap_result = self._run_agent("roadmap_creator_agent", roadmap_prompt)
            results["roadmap"] = roadmap_result
            st.session_state.learning_roadmap = roadmap_result
            
            status.update(label="Roadmap created!", state="complete")
        
        return results
    
    def find_resources(self):
        """
        Find and recommend learning resources for the topic.
        
        Returns:
            dict: Resource recommendations
        """
        results = {}
        
        with st.status("Finding learning resources...", expanded=True) as status:
            status.update(label="Searching for resources...", state="running")
            
            resource_prompt = self._format_prompt(
                self.config["prompts"]["resource_finding"]["base"],
                topic=self.topic,
                learning_goal=self.learning_goal,
                knowledge_level=self.knowledge_level,
                learning_style=self.learning_style
            )
            
            resource_result = self._run_agent("resource_finder_agent", resource_prompt)
            results["resources"] = resource_result
            st.session_state.learning_resources = resource_result
            
            status.update(label="Resources found!", state="complete")
        
        return results
    
    def _parse_quiz_json(self, raw_text: str) -> dict:
        """
        Parse quiz JSON from LLM response, handling common formatting issues.
        """
        # Try direct parse first
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        
        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # Last resort: return None so caller can fall back
        return None

    def generate_quiz(self, difficulty_level: str = "intermediate", 
                     focus_areas: str = "general", num_questions: int = 10):
        """
        Generate a structured interactive quiz.
        
        Returns:
            dict: Quiz data with parsed questions, or raw text fallback
        """
        results = {}
        
        with st.status("Generating quiz...", expanded=True) as status:
            status.update(label="🧠 AI is creating your quiz questions...", state="running")
            
            quiz_prompt = self._format_prompt(
                self.config["prompts"]["quiz_generation"]["base"],
                topic=self.topic,
                difficulty_level=difficulty_level,
                focus_areas=focus_areas,
                num_questions=num_questions
            )
            
            quiz_result = self._run_agent("quiz_generator_agent", quiz_prompt)
            
            # Parse the JSON response
            parsed = self._parse_quiz_json(quiz_result)
            
            if parsed and "questions" in parsed:
                results["quiz_data"] = parsed
                results["quiz"] = None  # No raw text needed
                status.update(label=f"✅ Quiz ready! {len(parsed['questions'])} questions generated", state="complete")
            else:
                # Fallback to raw text if JSON parsing fails
                results["quiz_data"] = None
                results["quiz"] = quiz_result
                status.update(label="Quiz ready! (text mode)", state="complete")
        
        return results

    def evaluate_quiz(self, questions: list, user_answers: dict) -> dict:
        """
        Evaluate quiz answers and provide AI-powered feedback.
        
        Args:
            questions: List of question dicts from the quiz
            user_answers: Dict mapping question id to user's selected answer letter
            
        Returns:
            dict: Evaluation results with score, per-question results, and AI feedback
        """
        correct = 0
        total = len(questions)
        detailed_results = []
        
        for q in questions:
            qid = q["id"]
            user_answer = user_answers.get(str(qid), "")
            is_correct = user_answer == q["correct_answer"]
            if is_correct:
                correct += 1
            
            detailed_results.append({
                "id": qid,
                "question": q["question"],
                "user_answer": user_answer,
                "user_answer_text": q["options"].get(user_answer, "Not answered"),
                "correct_answer": q["correct_answer"],
                "correct_answer_text": q["options"].get(q["correct_answer"], ""),
                "is_correct": is_correct,
                "explanation": q.get("explanation", ""),
                "concept": q.get("concept", "")
            })
        
        percentage = round((correct / total) * 100) if total > 0 else 0
        
        # Build detailed results text for the AI evaluator
        results_text = ""
        for r in detailed_results:
            status = "✅ CORRECT" if r["is_correct"] else "❌ INCORRECT"
            results_text += f"\nQ{r['id']}: {r['question']}\n"
            results_text += f"  Student answered: {r['user_answer']} ({r['user_answer_text']}) - {status}\n"
            if not r["is_correct"]:
                results_text += f"  Correct answer: {r['correct_answer']} ({r['correct_answer_text']})\n"
            results_text += f"  Concept: {r['concept']}\n"
        
        # Get AI-powered feedback using the evaluator agent
        ai_feedback = None
        try:
            eval_prompt = self._format_prompt(
                self.config["prompts"]["answer_evaluation"]["base"],
                topic=self.topic,
                knowledge_level=self.knowledge_level,
                score=correct,
                total=total,
                percentage=percentage,
                detailed_results=results_text
            )
            ai_feedback = self._run_agent("answer_evaluator_agent", eval_prompt)
        except Exception:
            ai_feedback = None
        
        return {
            "score": correct,
            "total": total,
            "percentage": percentage,
            "detailed_results": detailed_results,
            "ai_feedback": ai_feedback
        }
    
    def get_tutoring(self, student_question: str, context: str = ""):
        """
        Get tutoring help on a specific question.
        
        Args:
            student_question (str): The student's question
            context (str): Additional context
            
        Returns:
            str: Tutoring response
        """
        tutor_prompt = self._format_prompt(
            self.config["prompts"]["tutoring"]["base"],
            student_question=student_question,
            context=context,
            knowledge_level=self.knowledge_level
        )
        
        return self._run_agent("tutor_agent", tutor_prompt)
    
    def initialize_rag(self, collection_name: str = "study_materials"):
        """
        Initialize RAG helper for document-based learning.
        
        Args:
            collection_name (str): Name for the document collection
        """
        self.rag_helper = RAGHelper(collection_name=collection_name)
    
    def add_document_to_rag(self, file_path: str, file_type: str = "pdf") -> bool:
        """
        Add a document to the RAG knowledge base.
        
        Args:
            file_path (str): Path to the document
            file_type (str): Type of document ("pdf" or "text")
            
        Returns:
            bool: Success status
        """
        if not self.rag_helper:
            self.initialize_rag()
        
        if file_type == "pdf":
            return self.rag_helper.load_pdf(file_path)
        elif file_type == "text":
            return self.rag_helper.load_text(file_path)
        return False
    
    def query_documents(self, question: str, k: int = 4):
        """
        Query the uploaded documents using RAG.
        
        Args:
            question (str): The question to ask
            k (int): Number of relevant chunks to retrieve
            
        Returns:
            str: Answer based on documents
        """
        if not self.rag_helper:
            return "No documents have been uploaded yet. Please upload study materials first."
        
        # Retrieve relevant context
        relevant_docs = self.rag_helper.query(question, k=k)
        
        if not relevant_docs:
            return "I couldn't find relevant information in your uploaded documents. Please try rephrasing your question or upload more materials."
        
        # Combine context
        context = "\n\n".join(relevant_docs)
        
        # Use RAG tutor agent
        rag_prompt = self._format_prompt(
            self.config["prompts"]["rag_query"]["base"],
            question=question,
            context=context
        )
        
        return self._run_agent("rag_tutor_agent", rag_prompt)

    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the RAG knowledge base.
        
        Returns:
            int: Number of documents
        """
        if not self.rag_helper:
            return 0
        return self.rag_helper.get_document_count()
    
    def clear_documents(self) -> bool:
        """
        Clear all documents from the RAG knowledge base.
        
        Returns:
            bool: Success status
        """
        if not self.rag_helper:
            return False
        return self.rag_helper.clear_database()

import streamlit as st
from dotenv import load_dotenv
from agent_handler import StudyAssistantHandler
from config import ConfigManager
import os

# Load environment variables
# Load environment variables from the directory of app.py
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

# Page configuration
st.set_page_config(page_title="AI Study Assistant", layout="wide", page_icon="📚")
st.title("📚 Multi-Agent AI Study Assistant")
st.markdown("Personalized learning with AI agents - Analysis, Roadmaps, Quizzes & RAG-powered Tutoring!")

# Initialize config manager
config_manager = ConfigManager()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Provider selection - Gemini first (generous free tier), Groq as fallback
    provider = st.selectbox(
        "AI Provider",
        ["gemini", "groq", "openai"],
        index=0,
        help="Gemini has a generous free tier (recommended). Groq is used as automatic fallback."
    )
    
    # Auto-set API keys from .env (no manual entry needed)
    env_groq = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
    env_gemini = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")

    if env_groq:
        os.environ["GROQ_API_KEY"] = env_groq
    if env_gemini:
        os.environ["GEMINI_API_KEY"] = env_gemini
        os.environ["GOOGLE_API_KEY"] = env_gemini

    if provider == "gemini":
        selected_model = "gemini-3.8-flash"
        st.caption(f"✨ Model: **`{selected_model}`**")
        
        api_key_input = st.text_input(
            "Gemini API Key",
            value=env_gemini,
            type="password",
            help="Get your free key from https://aistudio.google.com"
        )
        if api_key_input:
            clean_key = api_key_input.strip().strip('"').strip("'")
            os.environ["GEMINI_API_KEY"] = clean_key
            os.environ["GOOGLE_API_KEY"] = clean_key

        groq_fallback_input = st.text_input(
            "Groq API Key (Auto-fallback on limit)",
            value=env_groq,
            type="password",
            help="Used automatically if Gemini rate limit is reached"
        )
        if groq_fallback_input:
            clean_groq = groq_fallback_input.strip().strip('"').strip("'")
            os.environ["GROQ_API_KEY"] = clean_groq
    elif provider == "groq":
        selected_model = "qwen/qwen3.8-27b"
        st.caption(f"⚡ Model: **`{selected_model}`**")
        
        api_key_input = st.text_input(
            "Groq API Key",
            value=env_groq,
            type="password",
            help="Get your free key from https://console.groq.com (starts with gsk_)"
        )
        if api_key_input:
            clean_key = api_key_input.strip().strip('"').strip("'")
            os.environ["GROQ_API_KEY"] = clean_key

        # Show Gemini fallback key
        gemini_fallback_input = st.text_input(
            "Gemini API Key (Auto-fallback on limit)",
            value=env_gemini,
            type="password",
            help="Used automatically if Groq rate limit is reached"
        )
        if gemini_fallback_input:
            clean_gemini = gemini_fallback_input.strip().strip('"').strip("'")
            os.environ["GEMINI_API_KEY"] = clean_gemini
            os.environ["GOOGLE_API_KEY"] = clean_gemini
    elif provider == "openai":
        selected_model = "gpt-4o"
        st.caption(f"🤖 Model: **`{selected_model}`**")
        
        env_openai = os.getenv("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        default_openai = env_openai if env_openai and not env_openai.startswith("your_") else ""
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=default_openai,
            type="password",
            help="Get your API key from https://platform.openai.com (starts with sk-)"
        )
        if api_key_input:
            clean_key = api_key_input.strip().strip('"').strip("'")
            os.environ["OPENAI_API_KEY"] = clean_key
    else:
        selected_model = "gemini-3.8-flash"
        st.caption(f"✨ Model: **`{selected_model}`**")






    
    st.divider()
    
    st.subheader("📖 About")
    st.markdown("""
    This AI Study Assistant uses multiple specialized agents to:
    - **Analyze** your learning needs
    - **Create** personalized roadmaps
    - **Generate** custom quizzes
    - **Tutor** you with RAG assistance
    - **Find** the best learning resources
    
    **Gemini**: Free tier via Google AI Studio (recommended)  
    **Groq**: Free, ultra-fast inference  
    **OpenAI**: Requires paid API key

    """)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "subject_category" not in st.session_state:
    st.session_state.subject_category = None
if "topic" not in st.session_state:
    st.session_state.topic = None
if "knowledge_level" not in st.session_state:
    st.session_state.knowledge_level = None
if "learning_goal" not in st.session_state:
    st.session_state.learning_goal = None
if "time_available" not in st.session_state:
    st.session_state.time_available = None
if "learning_style" not in st.session_state:
    st.session_state.learning_style = None
if "student_analysis" not in st.session_state:
    st.session_state.student_analysis = None
if "learning_roadmap" not in st.session_state:
    st.session_state.learning_roadmap = None
if "learning_resources" not in st.session_state:
    st.session_state.learning_resources = None
if "handler" not in st.session_state:
    st.session_state.handler = None
if "uploaded_files_count" not in st.session_state:
    st.session_state.uploaded_files_count = 0

# Step 1: Choose Subject Category
if st.session_state.step == 1:
    st.header("Step 1: Choose Your Subject Category")
    
    categories = config_manager.get_all_subject_categories()
    category_display = {
        "programming": "💻 Programming",
        "mathematics": "🔢 Mathematics",
        "science": "🔬 Science",
        "languages": "🌍 Languages",
        "business": "💼 Business",
        "test_preparation": "📝 Test Preparation"
    }
    
    cols = st.columns(3)
    for i, category in enumerate(categories):
        with cols[i % 3]:
            display_name = category_display.get(category, category.title())
            if st.button(display_name, use_container_width=True, key=f"cat_{category}"):
                st.session_state.subject_category = category
                st.session_state.step = 2
                st.rerun()

# Step 2: Enter Topic and Learning Details
elif st.session_state.step == 2:
    st.header("Step 2: Tell Us About Your Learning Goals")
    st.subheader(f"Category: {st.session_state.subject_category.title()}")
    
    # Show category info
    category_info = config_manager.get_subject_category_info(st.session_state.subject_category)
    if category_info:
        with st.expander("ℹ️ About this category"):
            st.write(f"**Topics include:** {', '.join(category_info.get('topics', []))}")
            st.write(f"**Typical Duration:** {category_info.get('typical_duration', 'Varies')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input(
            "What specific topic do you want to learn?",
            placeholder="e.g., Python for Data Science, Calculus, Spanish Grammar"
        )
        
        knowledge_levels = config_manager.get_all_knowledge_levels()
        knowledge_level = st.selectbox(
            "What's your current knowledge level?",
            knowledge_levels,
            format_func=lambda x: x.title()
        )
        
        # Show knowledge level info
        level_info = config_manager.get_knowledge_level_info(knowledge_level)
        if level_info:
            st.info(f"**{knowledge_level.title()}:** {level_info.get('description', '')}")
    
    with col2:
        learning_goal = st.text_area(
            "What do you want to achieve?",
            placeholder="e.g., Build a portfolio project, Pass an exam, Get job-ready",
            height=100
        )
        
        time_available = st.selectbox(
            "How much time can you dedicate?",
            ["1-2 hours per week", "3-5 hours per week", "6-10 hours per week", 
             "10+ hours per week", "Full-time (30+ hours per week)"]
        )
        
        learning_styles = config_manager.get_all_learning_styles()
        learning_style = st.selectbox(
            "What's your preferred learning style?",
            learning_styles,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Show learning style info
        style_info = config_manager.get_learning_style_info(learning_style)
        if style_info:
            st.info(f"**{learning_style.replace('_', ' ').title()}:** {style_info.get('description', '')}")
    
    col_back, col_next = st.columns([1, 5])
    with col_back:
        if st.button("← Back", type="secondary"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Create My Learning Plan", type="primary", 
                    disabled=not (topic and learning_goal)):
            st.session_state.topic = topic
            st.session_state.knowledge_level = knowledge_level
            st.session_state.learning_goal = learning_goal
            st.session_state.time_available = time_available
            st.session_state.learning_style = learning_style
            st.session_state.step = 3
            st.rerun()

# Step 3: Generate Analysis and Roadmap
elif st.session_state.step == 3:
    st.header("Creating Your Personalized Learning Plan")
    st.subheader(f"Topic: {st.session_state.topic}")
    
    # Validate API key
    if provider == "gemini":
        active_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        help_url = "https://aistudio.google.com"
    elif provider == "groq":
        active_key = os.getenv("GROQ_API_KEY", "").strip()
        help_url = "https://console.groq.com"
    elif provider == "openai":
        active_key = os.getenv("OPENAI_API_KEY", "").strip()
        help_url = "https://platform.openai.com"
    else:
        active_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        help_url = "https://aistudio.google.com"

    if not active_key or active_key.startswith("your_"):
        st.error(f"⚠️ **{provider.upper()} API Key is missing!**")
        st.info(f"Please enter your **{provider.upper()} API Key** in the sidebar on the left, or add it to your `.env` file to proceed.")
        st.markdown(f"👉 Get your {provider.title()} API key at [{help_url}]({help_url}).")
        if st.button("← Back to Step 2"):
            st.session_state.step = 2
            st.rerun()
    else:

        try:
            # Initialize or update handler if model/provider changed
            if (
                not st.session_state.handler
                or st.session_state.handler.model_name != selected_model
                or st.session_state.handler.provider != provider
            ):
                st.session_state.handler = StudyAssistantHandler(
                    topic=st.session_state.topic,
                    subject_category=st.session_state.subject_category,
                    knowledge_level=st.session_state.knowledge_level,
                    learning_goal=st.session_state.learning_goal,
                    time_available=st.session_state.time_available,
                    learning_style=st.session_state.learning_style,
                    model_name=selected_model,
                    provider=provider
                )

            
            # Analyze student
            if not st.session_state.student_analysis:
                analysis_results = st.session_state.handler.analyze_student()
            
            # Create roadmap
            if st.session_state.student_analysis and not st.session_state.learning_roadmap:
                roadmap_results = st.session_state.handler.create_roadmap(
                    st.session_state.student_analysis
                )
            
            # Find resources
            if st.session_state.learning_roadmap and not st.session_state.learning_resources:
                resource_results = st.session_state.handler.find_resources()
            
            # Move to dashboard when complete
            if (st.session_state.student_analysis and 
                st.session_state.learning_roadmap and 
                st.session_state.learning_resources):
                st.session_state.step = 4
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating learning plan: {e}")
            st.warning("Please verify that your API key is active and valid.")
            if st.button("🔄 Try Again"):
                st.session_state.student_analysis = None
                st.session_state.learning_roadmap = None
                st.session_state.learning_resources = None
                st.session_state.handler = None
                st.rerun()


# Step 4: Learning Dashboard
elif st.session_state.step == 4:
    st.header("🎯 Your Learning Dashboard")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Learning Roadmap", 
        "📚 Resources", 
        "❓ Quiz Generator",
        "🤖 AI Tutor",
        "📄 Document Q&A (RAG)"
    ])
    
    # Tab 1: Learning Roadmap
    with tab1:
        st.subheader("Your Personalized Learning Roadmap")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"**Topic:** {st.session_state.topic} | **Level:** {st.session_state.knowledge_level.title()}")
        with col2:
            if st.button("🔄 Regenerate Roadmap"):
                st.session_state.learning_roadmap = None
                roadmap_results = st.session_state.handler.create_roadmap(
                    st.session_state.student_analysis
                )
                st.rerun()
        
        st.markdown(st.session_state.learning_roadmap)
        
        with st.expander("📊 View Student Analysis"):
            st.markdown(st.session_state.student_analysis)
        
        st.download_button(
            label="📥 Download Roadmap",
            data=f"# Learning Roadmap: {st.session_state.topic}\n\n{st.session_state.learning_roadmap}",
            file_name=f"roadmap_{st.session_state.topic.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    
    # Tab 2: Resources
    with tab2:
        st.subheader("Recommended Learning Resources")
        
        if st.button("🔄 Find New Resources"):
            st.session_state.learning_resources = None
            resource_results = st.session_state.handler.find_resources()
            st.rerun()
        
        st.markdown(st.session_state.learning_resources)
        
        st.download_button(
            label="📥 Download Resources",
            data=f"# Learning Resources: {st.session_state.topic}\n\n{st.session_state.learning_resources}",
            file_name=f"resources_{st.session_state.topic.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    
    # Tab 3: Interactive Quiz
    with tab3:
        st.subheader("🎯 Interactive Quiz")
        
        # Initialize quiz session state
        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = None
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False
        if "quiz_evaluation" not in st.session_state:
            st.session_state.quiz_evaluation = None
        if "quiz_answers" not in st.session_state:
            st.session_state.quiz_answers = {}
        
        # Quiz generation controls
        col1, col2, col3 = st.columns(3)
        with col1:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["beginner", "intermediate", "advanced"],
                index=1
            )
        with col2:
            num_questions = st.slider("Number of Questions", 3, 15, 5)
        with col3:
            focus_areas = st.text_input(
                "Focus Areas (optional)",
                placeholder="e.g., loops, functions"
            )
        
        if st.button("🎲 Generate New Quiz", type="primary"):
            # Reset quiz state
            st.session_state.quiz_data = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_evaluation = None
            st.session_state.quiz_answers = {}
            st.session_state.current_quiz = None
            
            with st.spinner("Generating quiz..."):
                quiz_results = st.session_state.handler.generate_quiz(
                    difficulty_level=difficulty,
                    focus_areas=focus_areas if focus_areas else "general",
                    num_questions=num_questions
                )
                
                if quiz_results.get("quiz_data"):
                    st.session_state.quiz_data = quiz_results["quiz_data"]
                    st.session_state.current_quiz = None
                else:
                    # Fallback to text mode
                    st.session_state.quiz_data = None
                    st.session_state.current_quiz = quiz_results.get("quiz", "")
                
                st.rerun()
        
        # Display interactive quiz
        if st.session_state.quiz_data and not st.session_state.quiz_submitted:
            quiz = st.session_state.quiz_data
            title = quiz.get("quiz_title", f"Quiz on {st.session_state.topic}")
            questions = quiz.get("questions", [])
            
            st.markdown(f"### 📝 {title}")
            st.info(f"**{len(questions)} questions** • Answer all questions then click Submit")
            
            st.divider()
            
            # Render each question with radio buttons
            for i, q in enumerate(questions):
                qid = str(q["id"])
                diff_badge = {"easy": "🟢 Easy", "medium": "🟡 Medium", "hard": "🔴 Hard"}.get(
                    q.get("difficulty", "medium"), "🟡 Medium"
                )
                
                st.markdown(f"**Question {i+1}** &nbsp; {diff_badge}")
                st.markdown(f"**{q['question']}**")
                
                options = q.get("options", {})
                option_labels = [f"{key}) {val}" for key, val in options.items()]
                option_keys = list(options.keys())
                
                # Get previous selection if any
                prev_idx = None
                if qid in st.session_state.quiz_answers:
                    prev_answer = st.session_state.quiz_answers[qid]
                    if prev_answer in option_keys:
                        prev_idx = option_keys.index(prev_answer)
                
                selected = st.radio(
                    f"Select your answer for Q{i+1}:",
                    option_labels,
                    index=prev_idx,
                    key=f"quiz_q_{qid}",
                    label_visibility="collapsed"
                )
                
                # Store the selected answer letter
                if selected:
                    selected_key = selected.split(")")[0].strip()
                    st.session_state.quiz_answers[qid] = selected_key
                
                st.divider()
            
            # Submit button
            all_answered = len(st.session_state.quiz_answers) == len(questions)
            
            col_submit, col_count = st.columns([1, 3])
            with col_submit:
                submit_disabled = not all_answered
                if st.button("✅ Submit Quiz", type="primary", disabled=submit_disabled):
                    with st.spinner("🧠 AI is evaluating your answers..."):
                        evaluation = st.session_state.handler.evaluate_quiz(
                            questions, st.session_state.quiz_answers
                        )
                        st.session_state.quiz_evaluation = evaluation
                        st.session_state.quiz_submitted = True
                        st.rerun()
            with col_count:
                answered = len(st.session_state.quiz_answers)
                total = len(questions)
                if not all_answered:
                    st.warning(f"⚠️ Answered {answered}/{total} questions. Answer all to submit.")
                else:
                    st.success(f"✅ All {total} questions answered. Ready to submit!")
        
        # Display quiz results
        elif st.session_state.quiz_submitted and st.session_state.quiz_evaluation:
            evaluation = st.session_state.quiz_evaluation
            score = evaluation["score"]
            total = evaluation["total"]
            percentage = evaluation["percentage"]
            
            # Score header with emoji
            if percentage >= 80:
                grade_emoji = "🏆"
                grade_color = "green"
                grade_text = "Excellent!"
            elif percentage >= 60:
                grade_emoji = "👍"
                grade_color = "orange"
                grade_text = "Good job!"
            else:
                grade_emoji = "💪"
                grade_color = "red"
                grade_text = "Keep practicing!"
            
            st.markdown(f"## {grade_emoji} Quiz Results: {grade_text}")
            
            # Score metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score}/{total}")
            with col2:
                st.metric("Percentage", f"{percentage}%")
            with col3:
                st.metric("Correct", f"{score}", delta=f"{score - (total - score)} vs incorrect")
            
            # Progress bar
            st.progress(percentage / 100)
            
            st.divider()
            
            # Per-question results
            st.markdown("### 📋 Question-by-Question Review")
            
            for r in evaluation["detailed_results"]:
                if r["is_correct"]:
                    with st.expander(f"✅ Q{r['id']}: {r['question'][:80]}...", expanded=False):
                        st.success(f"**Your answer:** {r['user_answer']}) {r['user_answer_text']}")
                        st.markdown(f"**Concept:** {r['concept']}")
                        st.markdown(f"**Explanation:** {r['explanation']}")
                else:
                    with st.expander(f"❌ Q{r['id']}: {r['question'][:80]}...", expanded=True):
                        st.error(f"**Your answer:** {r['user_answer']}) {r['user_answer_text']}")
                        st.success(f"**Correct answer:** {r['correct_answer']}) {r['correct_answer_text']}")
                        st.markdown(f"**Concept:** {r['concept']}")
                        st.markdown(f"**Explanation:** {r['explanation']}")
            
            # AI Feedback section
            if evaluation.get("ai_feedback"):
                st.divider()
                st.markdown("### 🤖 AI Performance Analysis")
                st.markdown(evaluation["ai_feedback"])
            
            # Action buttons
            st.divider()
            col_retry, col_new, col_download = st.columns(3)
            with col_retry:
                if st.button("🔄 Retry Same Quiz"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_evaluation = None
                    st.session_state.quiz_answers = {}
                    st.rerun()
            with col_new:
                if st.button("🎲 Generate New Quiz"):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_evaluation = None
                    st.session_state.quiz_answers = {}
                    st.session_state.current_quiz = None
                    st.rerun()
            with col_download:
                # Build download content
                download_text = f"# Quiz Results: {st.session_state.topic}\n"
                download_text += f"Score: {score}/{total} ({percentage}%)\n\n"
                for r in evaluation["detailed_results"]:
                    status = "✅" if r["is_correct"] else "❌"
                    download_text += f"{status} Q{r['id']}: {r['question']}\n"
                    download_text += f"   Your answer: {r['user_answer']}) {r['user_answer_text']}\n"
                    download_text += f"   Correct: {r['correct_answer']}) {r['correct_answer_text']}\n"
                    download_text += f"   Explanation: {r['explanation']}\n\n"
                if evaluation.get("ai_feedback"):
                    download_text += f"\n---\n\n## AI Feedback\n{evaluation['ai_feedback']}\n"
                
                st.download_button(
                    label="📥 Download Results",
                    data=download_text,
                    file_name=f"quiz_results_{st.session_state.topic.replace(' ', '_')}.md",
                    mime="text/markdown"
                )
        
        # Fallback: text-mode quiz (if JSON parsing failed)
        elif "current_quiz" in st.session_state and st.session_state.current_quiz:
            st.markdown(st.session_state.current_quiz)
            st.download_button(
                label="📥 Download Quiz",
                data=f"# Quiz: {st.session_state.topic}\n\n{st.session_state.current_quiz}",
                file_name=f"quiz_{st.session_state.topic.replace(' ', '_')}.md",
                mime="text/markdown"
            )
        else:
            st.markdown("👆 Configure your quiz settings above and click **Generate New Quiz** to start!")
    
    # Tab 4: AI Tutor
    with tab4:
        st.subheader("Ask Your AI Tutor")
        st.write("Get personalized explanations and help with your questions.")
        
        question = st.text_area(
            "What would you like help with?",
            placeholder="e.g., Can you explain recursion with an example?",
            height=100
        )
        
        context = st.text_input(
            "Additional context (optional)",
            placeholder="e.g., I'm working on a specific problem..."
        )
        
        if st.button("💬 Ask Tutor", type="primary", disabled=not question):
            with st.spinner("Thinking..."):
                response = st.session_state.handler.get_tutoring(question, context)
                st.session_state.tutor_response = response
        
        if "tutor_response" in st.session_state and st.session_state.tutor_response:
            st.markdown("### 🤖 Tutor Response:")
            st.markdown(st.session_state.tutor_response)
    
    # Tab 5: Document Q&A (RAG)
    with tab5:
        st.subheader("Upload Study Materials & Ask Questions")
        st.write("Upload PDFs or text files and ask questions about them using RAG technology.")
        
        # File upload section
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Upload study materials (PDF or TXT)",
                type=["pdf", "txt"],
                accept_multiple_files=True
            )
        
        with col2:
            if st.session_state.handler:
                doc_count = st.session_state.handler.get_document_count()
                st.metric("Documents Loaded", doc_count)
                
                if doc_count > 0 and st.button("🗑️ Clear All Documents"):
                    st.session_state.handler.clear_documents()
                    st.session_state.uploaded_files_count = 0
                    st.success("Documents cleared!")
                    st.rerun()
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save file temporarily
                temp_path = f"./temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Add to RAG
                file_type = "pdf" if uploaded_file.name.endswith(".pdf") else "text"
                success = st.session_state.handler.add_document_to_rag(temp_path, file_type)
                
                if success:
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.session_state.uploaded_files_count += 1
                else:
                    st.error(f"❌ Failed to load: {uploaded_file.name}")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        st.divider()
        
        # Question answering section
        if st.session_state.handler and st.session_state.handler.get_document_count() > 0:
            st.markdown("### 💡 Ask Questions About Your Documents")
            
            doc_question = st.text_area(
                "What would you like to know?",
                placeholder="e.g., What are the key concepts in chapter 3?",
                height=100,
                key="doc_question"
            )
            
            if st.button("🔍 Search Documents", type="primary", disabled=not doc_question):
                with st.spinner("Searching documents..."):
                    answer = st.session_state.handler.query_documents(doc_question)
                    st.session_state.rag_answer = answer
            
            if "rag_answer" in st.session_state and st.session_state.rag_answer:
                st.markdown("### 📖 Answer from Your Documents:")
                st.markdown(st.session_state.rag_answer)
        else:
            st.info("📤 Upload documents above to start asking questions!")
    
    # Bottom navigation
    st.divider()
    if st.button("🏠 Start New Learning Plan"):
        # Clear session state
        keys_to_clear = [k for k in st.session_state.keys() if k != "step"]
        for key in keys_to_clear:
            del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

if __name__ == "__main__":
    pass

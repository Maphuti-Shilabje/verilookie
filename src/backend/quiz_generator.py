import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
import uuid
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API with error handling
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    client = None

class QuizQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=10, max_length=500)
    options: List[str] = Field(..., min_items=2, max_items=6)
    correct_answer: str
    explanation: str = Field(..., min_length=10, max_length=1000)
    difficulty: int = Field(..., ge=1, le=5)
    topic: str = Field(..., min_length=1, max_length=100)
    tags: List[str] = Field(default_factory=list)
    time_limit: Optional[int] = Field(default=30, description="Time limit in seconds")
    
    @validator('correct_answer')
    def correct_answer_must_be_in_options(cls, v, values):
        if 'options' in values and v not in values['options']:
            raise ValueError('Correct answer must be one of the provided options')
        return v

class Quiz(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=5, max_length=200)
    questions: List[QuizQuestion] = Field(..., min_items=1, max_items=50)
    difficulty: int = Field(..., ge=1, le=5)
    topic: str = Field(..., min_length=1, max_length=100)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    estimated_time: int = Field(default=0, description="Estimated completion time in minutes")
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.estimated_time == 0:
            # Calculate estimated time based on questions
            self.estimated_time = max(1, len(self.questions) * 1)  # 1 minute per question minimum

class UserPerformance(BaseModel):
    user_id: str
    quiz_id: str
    score: int = Field(..., ge=0)
    total: int = Field(..., gt=0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    xp_earned: int = Field(..., ge=0)
    time_taken: Optional[int] = Field(None, description="Time taken in seconds")
    difficulty_level: int = Field(..., ge=1, le=5)
    
    @validator('score')
    def score_cannot_exceed_total(cls, v, values):
        if 'total' in values and v > values['total']:
            raise ValueError('Score cannot exceed total questions')
        return v

class QuizGenerationError(Exception):
    """Custom exception for quiz generation errors"""
    pass

class QuizGenerator:
    def __init__(self):
        self.client = client
        self._validate_client()
        
        # Enhanced prompt templates for different difficulty levels
        self.difficulty_prompts = {
            1: "Create beginner-level questions focusing on basic concepts and definitions",
            2: "Create easy questions with straightforward applications of concepts",
            3: "Create intermediate questions requiring understanding and analysis",
            4: "Create advanced questions involving synthesis and evaluation",
            5: "Create expert-level questions requiring deep analysis and critical thinking"
        }
        
        # Topic-specific enhancements
        self.topic_contexts = {
            "AI detection": "Focus on AI detection methods, deepfake identification, and digital media authenticity",
            "deepfakes": "Cover deepfake technology, detection methods, ethical implications, and real-world applications",
            "machine learning": "Include algorithms, model evaluation, overfitting, and practical applications",
            "cybersecurity": "Focus on threats, prevention methods, encryption, and security best practices"
        }
    
    def _validate_client(self):
        """Validate that the Gemini client is properly initialized"""
        if not self.client:
            raise QuizGenerationError("Gemini API client not initialized. Check your API key.")
    
    async def generate_quiz_questions(self, topic: str, difficulty: int, count: int = 5,
                                    question_types: List[str] = None,
                                    avoid_topics: List[str] = None) -> List[QuizQuestion]:
        """
        Generate enhanced quiz questions using Gemini AI.
        
        Args:
            topic: The subject area for the quiz
            difficulty: Difficulty level (1-5)
            count: Number of questions to generate
            question_types: Types of questions to include (e.g., ['multiple_choice', 'true_false'])
            avoid_topics: Subtopics to avoid in questions
            
        Returns:
            List of QuizQuestion objects
        """
        try:
            self._validate_inputs(topic, difficulty, count)
            
            # Enhanced prompt construction
            prompt = self._build_enhanced_prompt(topic, difficulty, count, question_types, avoid_topics)
            
            # Generate content with retry logic
            response = await self._generate_with_retry(prompt)
            
            # Parse and validate response
            questions = self._parse_and_validate_response(response, topic, difficulty)
            
            # Ensure we have the requested number of questions
            if len(questions) < count:
                logger.warning(f"Generated {len(questions)} questions instead of requested {count}")
                # Fill missing questions with enhanced defaults
                questions.extend(self._get_enhanced_default_questions(
                    topic, difficulty, count - len(questions), len(questions)
                ))
            
            return questions[:count]  # Ensure exact count
            
        except Exception as e:
            logger.error(f"Error generating quiz questions: {e}")
            return self._get_enhanced_default_questions(topic, difficulty, count)
    
    def _validate_inputs(self, topic: str, difficulty: int, count: int):
        """Validate input parameters"""
        if not topic or len(topic.strip()) == 0:
            raise ValueError("Topic cannot be empty")
        if not 1 <= difficulty <= 5:
            raise ValueError("Difficulty must be between 1 and 5")
        if not 1 <= count <= 50:
            raise ValueError("Question count must be between 1 and 50")
    
    def _build_enhanced_prompt(self, topic: str, difficulty: int, count: int,
                             question_types: List[str] = None,
                             avoid_topics: List[str] = None) -> str:
        """Build an enhanced prompt for better question generation"""
        
        difficulty_desc = self.difficulty_prompts.get(difficulty, "Create appropriate level questions")
        topic_context = self.topic_contexts.get(topic.lower(), f"Focus on {topic} concepts and applications")
        
        prompt = f"""
        Generate {count} high-quality multiple-choice quiz questions about {topic}.
        
        Requirements:
        - Difficulty level: {difficulty}/5 - {difficulty_desc}
        - Topic context: {topic_context}
        - Each question should test understanding, not just memorization
        - Ensure questions are clear, unambiguous, and educational
        - Include diverse question formats within multiple choice
        """
        
        if avoid_topics:
            prompt += f"\n- Avoid these subtopics: {', '.join(avoid_topics)}"
            
        if question_types:
            prompt += f"\n- Focus on these question types: {', '.join(question_types)}"
        
        prompt += """
        
        Format your response as valid JSON with this exact structure:
        {
            "questions": [
                {
                    "question": "Clear, specific question text",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "The exact correct option from above",
                    "explanation": "Detailed explanation of why this answer is correct and others are wrong",
                    "tags": ["relevant", "topic", "tags"]
                }
            ]
        }
        
        Guidelines:
        - Make incorrect options plausible but clearly wrong
        - Include common misconceptions as distractors
        - Provide comprehensive explanations
        - Use relevant tags for categorization
        """
        
        return prompt
    
    async def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        genai.types.Content(
                            role="user",
                            parts=[
                                genai.types.Part.from_text(text=prompt),
                            ],
                        ),
                    ],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=3000,
                        top_p=0.9,
                        top_k=40
                    )
                )
                
                if response and response.text:
                    return response.text
                else:
                    raise QuizGenerationError("Empty response from Gemini API")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise QuizGenerationError(f"Failed to generate content after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_and_validate_response(self, content: str, topic: str, difficulty: int) -> List[QuizQuestion]:
        """Parse and validate the AI response"""
        try:
            # Clean the content
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            quiz_data = json.loads(content)
            
            if "questions" not in quiz_data:
                raise ValueError("Response missing 'questions' field")
            
            # Convert to QuizQuestion objects with validation
            questions = []
            for i, q_data in enumerate(quiz_data["questions"]):
                try:
                    # Fix for correct_answer field - if it's a letter, map it to the actual option
                    correct_answer = q_data["correct_answer"]
                    options = q_data["options"]
                    
                    # If correct_answer is a single letter (A, B, C, D) and we have options, map it
                    if isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
                        index = ord(correct_answer.upper()) - ord('A')
                        if 0 <= index < len(options):
                            correct_answer = options[index]
                    
                    question = QuizQuestion(
                        question=q_data["question"],
                        options=options,
                        correct_answer=correct_answer,
                        explanation=q_data["explanation"],
                        difficulty=difficulty,
                        topic=topic,
                        tags=q_data.get("tags", []),
                        time_limit=self._calculate_time_limit(difficulty)
                    )
                    questions.append(question)
                except Exception as e:
                    logger.warning(f"Failed to create question {i}: {e}")
                    continue
                    
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise QuizGenerationError("Invalid JSON response from AI")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise QuizGenerationError(f"Failed to parse AI response: {e}")
    
    def _calculate_time_limit(self, difficulty: int) -> int:
        """Calculate appropriate time limit based on difficulty"""
        base_time = 30  # seconds
        time_multiplier = {1: 0.8, 2: 1.0, 3: 1.2, 4: 1.5, 5: 2.0}
        return int(base_time * time_multiplier.get(difficulty, 1.0))
    
    def _get_enhanced_default_questions(self, topic: str, difficulty: int, count: int, 
                                      start_index: int = 0) -> List[QuizQuestion]:
        """Generate enhanced default questions with better variety"""
        default_questions = []
        
        # Enhanced default question templates based on topic
        templates = self._get_question_templates(topic, difficulty)
        
        for i in range(count):
            template_index = (start_index + i) % len(templates)
            template = templates[template_index]
            
            # Ensure correct_answer is the actual text, not a letter
            correct_answer = template["correct_answer"]
            options = template["options"]
            
            # If correct_answer is a single letter (A, B, C, D) and we have options, map it
            if isinstance(correct_answer, str) and len(correct_answer) == 1 and correct_answer.isalpha():
                index = ord(correct_answer.upper()) - ord('A')
                if 0 <= index < len(options):
                    correct_answer = options[index]
            
            question = QuizQuestion(
                id=f"default_{uuid.uuid4()}",
                question=template["question"].format(topic=topic, index=i+1),
                options=options,
                correct_answer=correct_answer,
                explanation=template["explanation"].format(topic=topic),
                difficulty=difficulty,
                topic=topic,
                tags=["default", topic.lower().replace(" ", "_")],
                time_limit=self._calculate_time_limit(difficulty)
            )
            default_questions.append(question)
        
        return default_questions
    
    def _get_question_templates(self, topic: str, difficulty: int) -> List[Dict]:
        """Get question templates based on topic and difficulty"""
        if "AI" in topic or "deepfake" in topic.lower():
            return [
                {
                    "question": f"What is a key indicator of AI-generated content in {topic}?",
                    "options": ["Unusual eye movements", "Perfect symmetry", "Consistent lighting", "All of the above"],
                    "correct_answer": "All of the above",
                    "explanation": f"Multiple indicators can help identify AI-generated content in {topic}."
                },
                {
                    "question": f"Which technique is commonly used to detect deepfakes in {topic}?",
                    "options": ["Temporal consistency analysis", "Color histogram analysis", "Edge detection", "Noise reduction"],
                    "correct_answer": "Temporal consistency analysis",
                    "explanation": f"Temporal consistency analysis is effective for detecting deepfakes in {topic}."
                }
            ]
        
        # Generic templates for other topics
        return [
            {
                "question": f"What is a fundamental concept in {topic}?",
                "options": [f"Concept A", f"Concept B", f"Concept C", f"All concepts are correct"],
                "correct_answer": f"All concepts are correct",
                "explanation": f"Understanding multiple concepts is important in {topic}."
            }
        ]
    
    async def create_personalized_quiz(self, user_id: str, topic: str = "AI detection and deepfakes",
                                     previous_performance: Optional[List[UserPerformance]] = None,
                                     preferences: Optional[Dict[str, Any]] = None) -> Quiz:
        """
        Create an enhanced personalized quiz based on user's previous performance and preferences.
        """
        try:
            # Enhanced user level calculation
            difficulty = self._calculate_adaptive_difficulty(previous_performance)
            
            # Apply user preferences
            question_count = 5
            question_types = None
            avoid_topics = None
            
            if preferences:
                question_count = preferences.get('question_count', 5)
                question_types = preferences.get('question_types')
                avoid_topics = preferences.get('avoid_topics')
            
            # Generate questions with enhanced parameters
            questions = await self.generate_quiz_questions(
                topic, difficulty, question_count, question_types, avoid_topics
            )
            
            # Create enhanced quiz
            quiz = Quiz(
                title=self._generate_quiz_title(topic, difficulty),
                questions=questions,
                difficulty=difficulty,
                topic=topic,
                description=f"Personalized {topic} quiz tailored to your skill level",
                tags=[topic.lower().replace(" ", "_"), f"level_{difficulty}", "personalized"]
            )
            
            logger.info(f"Created personalized quiz for user {user_id}: {quiz.id}")
            return quiz
            
        except Exception as e:
            logger.error(f"Error creating personalized quiz: {e}")
            # Fallback to basic quiz generation
            questions = self._get_enhanced_default_questions(topic, 2, 5)
            return Quiz(
                title=f"Basic {topic} Quiz",
                questions=questions,
                difficulty=2,
                topic=topic,
                description=f"Basic quiz on {topic}",
                tags=[topic.lower().replace(" ", "_"), "basic", "fallback"]
            )
    
    def _calculate_adaptive_difficulty(self, previous_performance: Optional[List[UserPerformance]]) -> int:
        """Enhanced difficulty calculation with trend analysis"""
        if not previous_performance:
            return 2  # Start with medium-low difficulty
            
        # Sort by timestamp (most recent first)
        sorted_performance = sorted(
            previous_performance, 
            key=lambda x: x.timestamp, 
            reverse=True
        )
        
        # Weight recent performance more heavily
        weighted_scores = []
        for i, perf in enumerate(sorted_performance[:10]):  # Consider last 10 quizzes
            if perf.total > 0:
                accuracy = perf.score / perf.total
                weight = 0.9 ** i  # Exponential decay for older results
                weighted_scores.append((accuracy, weight))
        
        if not weighted_scores:
            return 2
            
        # Calculate weighted average
        total_weight = sum(weight for _, weight in weighted_scores)
        weighted_avg = sum(acc * weight for acc, weight in weighted_scores) / total_weight
        
        # Check for improvement trend
        if len(sorted_performance) >= 3:
            recent_avg = sum(p.score / p.total for p in sorted_performance[:3]) / 3
            older_avg = sum(p.score / p.total for p in sorted_performance[3:6]) / min(3, len(sorted_performance[3:6]))
            
            if recent_avg > older_avg + 0.1:  # Improving trend
                weighted_avg += 0.1
        
        # Map to difficulty level with more nuanced scaling
        if weighted_avg >= 0.95:
            return 5
        elif weighted_avg >= 0.85:
            return 4
        elif weighted_avg >= 0.70:
            return 3
        elif weighted_avg >= 0.50:
            return 2
        else:
            return 1
    
    def _generate_quiz_title(self, topic: str, difficulty: int) -> str:
        """Generate an engaging quiz title"""
        difficulty_names = {1: "Beginner", 2: "Easy", 3: "Intermediate", 4: "Advanced", 5: "Expert"}
        difficulty_name = difficulty_names.get(difficulty, "Mixed")
        
        title_templates = [
            f"{difficulty_name} {topic} Challenge",
            f"{topic} Mastery: {difficulty_name} Level",
            f"Test Your {topic} Skills - {difficulty_name}",
            f"{topic} Quiz: {difficulty_name} Edition"
        ]
        
        # Use hash for consistent selection based on topic
        hash_value = int(hashlib.md5(topic.encode()).hexdigest(), 16)
        return title_templates[hash_value % len(title_templates)]
    
    def calculate_xp_reward(self, quiz: Quiz, score: int, total: int, 
                          time_taken: Optional[int] = None) -> int:
        """
        Enhanced XP calculation with time bonus and streak multipliers.
        """
        if total == 0:
            return 0
            
        accuracy = score / total
        
        # Base XP calculation
        base_xp = quiz.difficulty * 15  # Increased base XP
        
        # Accuracy bonuses with more granular rewards
        if accuracy == 1.0:
            accuracy_bonus = 50  # Perfect score
        elif accuracy >= 0.9:
            accuracy_bonus = 30
        elif accuracy >= 0.8:
            accuracy_bonus = 20
        elif accuracy >= 0.7:
            accuracy_bonus = 10
        elif accuracy >= 0.6:
            accuracy_bonus = 5
        else:
            accuracy_bonus = 0
        
        # Time bonus (if completed quickly)
        time_bonus = 0
        if time_taken and quiz.estimated_time:
            expected_time = quiz.estimated_time * 60  # Convert to seconds
            if time_taken < expected_time * 0.7:  # Completed in less than 70% of expected time
                time_bonus = 10
            elif time_taken < expected_time * 0.8:
                time_bonus = 5
        
        # Question difficulty bonus
        question_bonus = sum(q.difficulty for q in quiz.questions) * 2
        
        total_xp = base_xp + accuracy_bonus + time_bonus + question_bonus
        
        # Minimum XP for participation (even if score is 0)
        participation_xp = 5
        
        return max(participation_xp, total_xp)

# Global instance with enhanced error handling
try:
    quiz_generator = QuizGenerator()
    logger.info("Quiz generator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize quiz generator: {e}")
    quiz_generator = None
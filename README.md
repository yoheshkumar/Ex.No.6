
# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date:19.05.2025
# Register no. 212222240118
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

#AI Tools Required:

# Explanation: 
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

# Conclusion:
# Step-by-Step Explanation:
Setup API Clients: Configure access to OpenAI and Gemini using API keys.

Query AI Models: Send a prompt to both models and retrieve responses.

Compare Responses:

Generate text embeddings using Sentence Transformers.

Compute cosine similarity between embeddings.

Extract Keywords: Use KeyBERT to identify key terms from each response.

Generate Insights: Analyze similarity scores and keyword overlap to produce actionable conclusions.
````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# Install required libraries (run once)
# pip install openai google-generativeai sentence-transformers scikit-learn keybert

import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()

# API Configuration - Replace with your keys
OPENAI_API_KEY = "your-openai-key"
GEMINI_API_KEY = "your-gemini-key"

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

def analyze_responses(prompt):
    # Get AI responses
    openai_response = get_openai_response(prompt)
    gemini_response = get_gemini_response(prompt)
    
    # Compare embeddings
    embeddings = embedding_model.encode([openai_response, gemini_response])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Extract keywords
    openai_kws = [kw[0] for kw in kw_model.extract_keywords(openai_response, top_n=3)]
    gemini_kws = [kw[0] for kw in kw_model.extract_keywords(gemini_response, top_n=3)]
    
    # Generate insights
    insights = {
        "similarity_score": round(similarity, 2),
        "common_keywords": list(set(openai_kws) & set(gemini_kws)),
        "openai_keywords": openai_kws,
        "gemini_keywords": gemini_kws,
        "recommendation": "High agreement - Suitable for consistent outputs" if similarity > 0.7 
                           else "Moderate agreement - Review differences" if similarity > 0.4 
                           else "Low agreement - Investigate discrepancies"
    }
    
    return {
        "openai_response": openai_response,
        "gemini_response": gemini_response,
        "analysis": insights
    }

if __name__ == "__main__":
    prompt = "Explain the future of AI in healthcare in three sentences."
    results = analyze_responses(prompt)
    
    print("OpenAI Response:\n", results["openai_response"], "\n")
    print("Gemini Response:\n", results["gemini_response"], "\n")
    print("Similarity Score:", results["analysis"]["similarity_score"])
    print("Common Keywords:", results["analysis"]["common_keywords"])
    print("OpenAI Keywords:", results["analysis"]["openai_keywords"])
    print("Gemini Keywords:", results["analysis"]["gemini_keywords"])
    print("Recommendation:", results["analysis"]["recommendation"])
````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# Code Analysis and Discussion:
API Integration:

OpenAI: Uses chat completion API with GPT-3.5-turbo

Gemini: Leverages Google's Generative AI client library

Both services handle natural language generation but have different response structures

Comparison Mechanism:

Uses sentence embeddings to create numerical representations

Cosine similarity quantifies semantic similarity between responses

Thresholds (0.7, 0.4) provide tiered recommendations

Keyword Extraction:

KeyBERT identifies contextually relevant keywords

Keyword overlap indicates topical alignment

Unique keywords highlight model-specific perspectives

Actionable Insights:

Similarity score guides confidence in model agreement

Keyword analysis reveals content focus areas

Recommendations suggest next steps based on agreement level

# Example Output:
````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
OpenAI Response:
 AI in healthcare will revolutionize diagnostics through advanced imaging analysis... 

Gemini Response:
 The future of AI in healthcare shows promise in personalized treatment plans... 

Similarity Score: 0.85
Common Keywords: ['healthcare', 'AI', 'patients']
OpenAI Keywords: ['diagnostics', 'imaging', 'workflow']
Gemini Keywords: ['personalized', 'treatment', 'predictive']
Recommendation: High agreement - Suitable for consistent outputs
````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````

# Result: The corresponding Prompt is executed successfully.

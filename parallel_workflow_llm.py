from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(temperature=0, api_key=groq_api_key, model="llama-3.1-8b-instant")




class EvaluationSchema(BaseModel):

    feedback: str = Field(description='Detailed feedbackfor the essay')
    score: float = Field(description='Score out of 10', ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)



essay = """Pakistan in the Age of AI
As the world enters a transformative era defined by artificial intelligence (AI), Pakistan stands at a critical juncture — one where it can either emerge as a global leader in AI innovation or risk falling behind in the technology race. The age of AI brings with it immense promise as well as unprecedented challenges, and how Pakistan navigates this landscape will shape its socio-economic and geopolitical future.

Pakistan's strengths in the AI domain are rooted in its vast pool of talented engineers, a rapidly growing IT industry, and an emerging startup ecosystem. With over 300,000 STEM graduates annually and a rising base of AI researchers, Pakistan possesses the intellectual capital required to build cutting-edge AI systems. Institutions like NUST, LUMS, FAST, and COMSATS have begun fostering AI research, while private players such as Systems Limited, NETSOL, and 10Pearls are integrating AI into their global services. In 2021, the government launched the National Artificial Intelligence Policy (2021-2025) with a focus on digital transformation, aiming to leverage AI in healthcare, agriculture, education, and smart governance.

One of the most promising applications of AI in Pakistan lies in agriculture, where predictive analytics can guide farmers on optimal sowing times, weather forecasts, and pest control—especially vital in a country where 42% of the workforce depends on farming. In healthcare, AI-powered diagnostics can help address Pakistan’s severe doctor-patient ratio crisis, particularly in rural and underserved areas. Educational platforms like Taleemabad and Sabaq are increasingly using AI to personalize learning paths, while smart governance tools are helping improve public service delivery, tax collection, and fraud detection.

However, the path to AI-led growth is riddled with challenges. Chief among them is the digital divide. While urban centers like Karachi, Lahore, and Islamabad may embrace AI-driven solutions, rural Pakistan continues to struggle with basic internet access and digital literacy. The risk of job displacement due to automation also looms large, especially for low-skilled workers in textiles, retail, and informal sectors. Without effective skilling and re-skilling programs, AI could exacerbate existing socio-economic inequalities.

Another pressing concern is data privacy and ethics. As AI systems rely heavily on vast datasets, ensuring that personal data is used transparently and responsibly becomes vital. Pakistan is still developing its data protection framework through the Personal Data Protection Bill, and in the absence of a strong regulatory environment, AI systems may risk misuse or bias.

To harness AI responsibly, Pakistan must adopt a multi-stakeholder approach involving the government, academia, industry, and civil society. Policies should promote open datasets, encourage responsible innovation, and ensure ethical AI practices. There is also a need for international collaboration, particularly with countries leading in AI research, to gain strategic advantage and ensure interoperability in global systems.

Pakistan’s youth bulge, when paired with responsible AI adoption, can unlock massive economic growth, improve governance, and uplift marginalized communities. But this vision will only materialize if AI is seen not merely as a tool for automation, but as an enabler of human-centered development.

In conclusion, Pakistan in the age of AI is a story in the making — one of opportunity, responsibility, and transformation. The decisions we make today will not just determine Pakistan’s AI trajectory, but also its future as an inclusive, equitable, and innovation-driven society."""


prompt = """
Evaluate the language quality of the following essay.
Return your answer in JSON format with exactly two keys:
- feedback: (string)
- score: (integer between 0 and 10)

Essay:
{essay}
""".format(essay=essay)

# result = structured_model.invoke(prompt).feedback
# print(result)


class UPSCState(TypedDict):

    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float




def evaluate_language(state: UPSCState):
    prompt = f"""
You are an expert in English language and grammar. Evaluate ONLY the **language quality** of the essay.

Focus on:
1. Grammar & Syntax (0–10)
2. Vocabulary & Word Choice (0–10)
3. Sentence Fluency & Style (0–10)

Be strict. Do not consider content depth or structure.

Return strict JSON:
{{
  "feedback": "Detailed feedback on grammar, vocabulary, and fluency only.",
  "score": <average of 3 scores, rounded to nearest integer>
}}

Essay:
{state['essay']}
"""
    output = structured_model.invoke(prompt)
    return {
        'language_feedback': output.feedback,
        'individual_scores': [output.score]
    }



def evaluate_analysis(state: UPSCState):
    prompt = f"""
You are a UPSC examiner evaluating **analytical depth**.

Focus on:
1. Logical Reasoning & Argument Strength (0–10)
2. Use of Relevant Examples & Data (0–10)
3. Originality & Insight (0–10)

Be strict. Ignore grammar and structure.

Return strict JSON:
{{
  "feedback": "Feedback on logic, examples, and depth of analysis only.",
  "score": <average of 3 scores, rounded to nearest integer>
}}

Essay:
{state['essay']}
"""
    output = structured_model.invoke(prompt)
    return {
        'analysis_feedback': output.feedback,
        'individual_scores': [output.score]
    }



def evaluate_thought(state: UPSCState):
    prompt = f"""
You are an expert in essay structure and clarity.

Focus on:
1. Introduction & Conclusion Effectiveness (0–10)
2. Paragraph Flow & Transitions (0–10)
3. Overall Organization & Readability (0–10)

Be strict. Ignore grammar and content depth.

Return strict JSON:
{{
  "feedback": "Feedback on structure, flow, and clarity only.",
  "score": <average of 3 scores, rounded to nearest integer>
}}

Essay:
{state['essay']}
"""
    output = structured_model.invoke(prompt)
    return {
        'clarity_feedback': output.feedback,
        'individual_scores': [output.score]
    }



def final_evaluation(state: UPSCState):

    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback - {state["language_feedback"]} \n depth of analysis feedback - {state["analysis_feedback"]} \n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}



graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()





essay2 = """Pakistan and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). Pakistan also want become big in this AI thing. If work hard, Pakistan can go top. But if no careful, Pakistan go back.

Pakistan have many good. We have smart student, many engine-ear, and good IT peoples. Big company like NETSOL, Systems Limited, 10Pearls already use AI. Government also do program “National AI Policy”. It want AI in farm, doctor place, school and office work.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big bigGGGG data. Who take care? Pakistan still make data rule. If no strong rule, AI do bad.

Pakistan must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If Pakistan use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in Pakistan have many hope and many danger. We must go right road. AI must help all people, not only some. Then Pakistan grow big and world say "good job Pakistan"."""



intial_state = {
    'essay': essay2
}

result = workflow.invoke(intial_state)
print(result)




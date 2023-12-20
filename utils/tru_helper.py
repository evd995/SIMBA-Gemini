import streamlit as st
from trulens_eval import Tru, Feedback, TruLlama, OpenAI as fOpenAI
from trulens_eval.feedback import Groundedness
import openai
import numpy as np

openai.api_key = st.secrets["OPENAI_API_KEY"]
tru = Tru()

def build_tru_recorder(agent):
    provider = fOpenAI()
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text
    f_qs_relevance = (
        Feedback(provider.qs_relevance_with_cot_reasons,
                name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    grounded = Groundedness(groundedness_provider=provider)
    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons,
                name="Groundedness"
                )
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    tru_recorder = TruLlama(
        agent,
        app_id="Students Agent",
        feedbacks=[
            f_qa_relevance,
            f_qs_relevance,
            f_groundedness
        ]
    )
    return tru_recorder
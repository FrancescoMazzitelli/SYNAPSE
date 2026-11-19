"""Ablation study tooling for the SYNAPSE survey pipeline.

Everything here imports the production modules at the repository root and only
adds what the ablation needs: configurable memory, an external run spec and a
driver that composes the existing pieces (preprocess -> synthesize ->
gp_survey_engine -> postprocess).
"""

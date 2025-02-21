import math
import re

import numexpr
from langchain_core.tools import BaseTool, tool


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


class ModuLabsNotionRetrieverTool(BaseTool):

    name: str = "ModuLabsNotionRetrieverTool"
    description: str = "Retrieves information from Notion using the ModuLabs API."

    def __init__(self, **kwargs):
        from langchain_community.document_loaders import NotionDirectoryLoader
        super().__init__(**kwargs)

    def _run(self, *args, **kwargs):
        return super()._run(*args, **kwargs)

    def _arun(self, *args, **kwargs):
        return super()._arun(*args, **kwargs)

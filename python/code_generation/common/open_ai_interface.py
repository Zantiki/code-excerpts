from src.common.json_utils import read_json
from src.common.definitions import CREDENTIALS
from abc import abstractmethod
import openai

debug = False
openai.api_key = read_json(CREDENTIALS)["open_ai"]

class OpenAIInterface():

    stop_characters = "\n\n"
    @staticmethod
    @abstractmethod
    def _MODEL():
        pass

    def parse_response(self, response: dict):
        responses_key = "choices"
        responses = []
        if responses_key in response and response[responses_key]:
            for item in response[responses_key]:
                text_response = item["text"]

                if text_response:
                    responses.append(text_response)
        return responses

    def prompt(self, prompt: str):

        if debug:
            return self.dummy_response()
        response = openai.Completion.create(
                      model=self._MODEL(),
                      prompt=prompt,
                      max_tokens=100,
                      temperature=0
                    )

        return self.parse_response(response)

    def dummy_response(self):
        dummy_dict = {
                  "choices": [
                    {
                      "finish_reason": "stop",
                      "index": 0,
                      "logprobs": None,
                      "text": f"{self.stop_characters}import tu_madre;\nconsole.log(\"Hello World!\");"
                    }
                  ],
                  "created": 1678536222,
                  "id": "cmpl-6ssEooSLqebJt7Rg8BcHT1E6QP208",
                  "model": self._MODEL(),
                  "object": "text_completion",
                  "usage": {
                    "completion_tokens": 10,
                    "prompt_tokens": 5,
                    "total_tokens": 15
                  }
                }
        return self.parse_response(dummy_dict)


class CodeGeneratorInterface(OpenAIInterface):

    @staticmethod
    @abstractmethod
    def _MODEL():
        return "text-davinci-003"


class CodeEditorInterface(CodeGeneratorInterface):
    @staticmethod
    @abstractmethod
    def _MODEL():
        return "text-davinci-edit-001"

    def edit(self, text_to_change, instruction):
        if debug:
            return self.dummy_response()
        response = openai.Edit.create(
            model=self._MODEL(),
            input=text_to_change,
            instruction=instruction
        )
        return self.parse_response(response)
def test_create():
    test_prompt = "write a python function that takes a js file as input and merges the javascript imports"
    ai = CodeGeneratorInterface()
    response = ai.prompt(test_prompt)
    for response in response:
        print(response)

def test_edit():
    test_input = "const root = ReactDOM.createRoot(document.getElementById('root'));root.render(<React.StrictMode><App /></React.StrictMode>);"
    # test_instruction = "Expand the following Dom to include a component called Test"
    test_instruction = "Add component Test into root render"
    ai = CodeEditorInterface()
    response = ai.edit(test_input, test_instruction)
    for response in response:
        print(response)

def main():
    test_create()

if __name__ == "__main__":
    main()
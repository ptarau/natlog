import os
import webbrowser
import openai
import tiktoken
from natlog import Natlog, natprogs, lconsult

openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatMind:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.mem = []
        self.answers = [None]
        self.add_content('system', 'You are a helpful assistant.')
        self.toks=0

    def add_content(self, role, content):
        d = dict(role=role, content=content)
        self.mem.append(d)

    def __repr__(self):
        return str(self.mem)+"\n\n"+str(self.answers)+"\n\n"+f'tokens: {self.toks}\n'

    def ask(self, quest):
        self.add_content('user', quest)

        r = openai.ChatCompletion.create(
            model=self.model,
            messages=self.mem
        )
        result = r['choices'][0]['message']['content']
        result=result.split('\n')[-1]
        self.answers.append(result)
        t=r['usage']['total_tokens']
        self.toks+=t
        return result


def test_deepchat():
    cm = ChatMind()
    answer=cm.ask('What was the warmest temperature measured on Mars?')
    answer = cm.ask('The same question for planet Venus?')
    answer = cm.ask('And what about dwarf planet Ceres?')

    print(cm)



if __name__ == "__main__":
    test_deepchat()

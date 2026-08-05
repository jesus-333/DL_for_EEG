from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion

class DynamicCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # called fresh on every keystroke — build the list however you want
        options = my_lookup(text)   # DB query, API call, filtered list, etc.

        for opt in options:
            yield Completion(
                opt,
                start_position=-len(text),  # replace what's typed so far
                display=opt,                # what shows in the menu
                display_meta="detail",      # optional right-hand annotation
            )

def my_lookup(text):
    data = ['apple', 'apricot', 'banana', 'cherry', 'date']
    return [d for d in data if text.lower() in d.lower()]

text = prompt('> ', completer=DynamicCompleter(), complete_while_typing=True)
print(text)

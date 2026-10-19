import yaml
import os
from jinja2 import Template

class PromptManager:
    def __init__(self, prompt_path):
        self.prompt_path = prompt_path
        self.config = self._load_config()
        
    def _load_config(self):
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def render(self, template_str, **kwargs):
        """Render a template string with Jinja2."""
        if not template_str:
            return ""
        template = Template(template_str)
        return template.render(**kwargs)

    def get_messages(self, stage, **kwargs):
        """
        Build the messages list for a given stage.
        stage: 'planning', 'summary', 'step', 'final_answer'
        kwargs: template variables such as task, tools, tool_functions_json, etc.
        """
        messages = []
        
        # 1. Planning stage
        if stage == 'planning':
            # Planning has its own system prompt ("You are a world-class planning expert...")
            sys_tmpl = self.config['planning']['initial_plan']
            user_tmpl = self.config['planning']['task_input']
            
            messages.append({"role": "system", "content": self.render(sys_tmpl, **kwargs)})
            messages.append({"role": "user", "content": self.render(user_tmpl, **kwargs)})

        # 2. Step (execution) stage
        elif stage == 'step':
            sys_tmpl = self.config['system_prompt']

            user_tmpl = self.config['step']['pre_messages']
            
            messages.append({"role": "system", "content": self.render(sys_tmpl, **kwargs)})
            messages.append({"role": "user", "content": self.render(user_tmpl, **kwargs)})
            

        # 3. Summary stage (reflection and state update)
        elif stage == 'summary':
            # pre_messages acts as the system instruction, post_messages as the user instruction
            sys_tmpl = self.config['summary']['update_pre_messages']
            user_tmpl = self.config['summary']['update_post_messages']
            
            messages.append({"role": "system", "content": self.render(sys_tmpl, **kwargs)})
            # The summary stage usually also needs history; only the user instruction is returned here
            messages.append({"role": "user", "content": self.render(user_tmpl, **kwargs)})

        # 4. Final answer stage
        elif stage == 'final_answer':
            # pre_messages describes the context, post_messages holds the instruction;
            # they are typically merged into one user message, or used as system + user.
            # Following the YAML layout, pre is used as system and post as user.
            sys_tmpl = self.config['final_answer']['pre_messages']
            user_tmpl = self.config['final_answer']['post_messages']
            
            messages.append({"role": "system", "content": self.render(sys_tmpl, **kwargs)})
            messages.append({"role": "user", "content": self.render(user_tmpl, **kwargs)})
            
        else:
            raise ValueError(f"Unknown stage: {stage}")

        return messages
import subprocess
import sys
from time import sleep
import re
from src.common.open_ai_interface import CodeGeneratorInterface, CodeEditorInterface
from src.common.definitions import CONTENT_GENERATION_ROOT, BIN_DIR, CONTENT_GENERATION_CONFIG


def merge_js_imports(content):

    # find all the imports
    imports = re.findall(r'import\s+(.*)\s+from\s+[\'"](.*)[\'"]', content)

    # create a dictionary to store the imports
    imports_dict = {}

    # loop through the imports
    for i in imports:
        # split the import into its components
        parts = i[1].split('/')
        # get the last part of the import
        last_part = parts[-1]
        # check if the last part is already in the dictionary
        if last_part in imports_dict:
            # if it is, append the import to the existing list
            imported = i[0].split(",")
            imported_cleaned = [comp.replace("{", "").replace("}", "") for comp in imported]
            imports_dict[last_part] += imported_cleaned
        else:
            # if it isn't, create a new list with the import
            imported = i[0].split(",")
            imported_cleaned = [comp.replace("{", "").replace("}", "") for comp in imported]
            imports_dict[last_part] = imported_cleaned

    # loop through the dictionary
    for k, v in imports_dict.items():
        # if there is more than one import for the same module
        if len(v) > 1:
            # replace all the imports with a single import
            content = content.replace(v[0], 'import { ' + ', '.join(v) + ' } from \'' + k + '\'')
            # remove the other imports
            for i in v[1:]:
                content = content.replace(i, '')

    # write the new content to the file
    return content


def import_cleanup(lines):
    # find all the imports
    imports = re.findall(r'import\s+(.*)\s+from\s+[\'"](.*)[\'"]', lines)

    # create a dictionary to store the imports
    imports_dict = {}

    # loop through the imports
    for i in imports:
        # split the import into its components
        parts = i[1].split('/')
        # get the last part of the import
        last_part = parts[-1]
        # check if the last part is already in the dictionary
        if last_part in imports_dict:
            # if it is, append the import to the existing list
            imported = i[0].split(",")
            imported_cleaned = [comp.replace("{", "").replace("}", "").strip() for comp in imported]
            imports_cleaned = [comp for comp in imported_cleaned if comp not in imports_dict[last_part]]
            imports_dict[last_part] += imports_cleaned
        else:
            # if it isn't, create a new list with the import
            imported = i[0].split(",")
            imported_cleaned = [comp.replace("{", "").replace("}", "") for comp in imported]
            imports_dict[last_part] = imported_cleaned

    import_lines =  ""
    for k, v in imports_dict.items():
        # if there is more than one import for the same module
        import_lines += 'import { ' + ', '.join(v) + ' } from \'./' + k + '\'\n'

    # write the new content to the file
    return import_lines


class WebappCodeGenerator:
    webapp_working_dir = CONTENT_GENERATION_ROOT
    create_key = "inserts"
    edits_key = "edits"

    def __init__(self, webapp_name: str, design_spec_file: str):
        self.webapp_name = webapp_name

        self.creation_ai_interface = CodeGeneratorInterface()
        self.edit_ai_interface = CodeEditorInterface()

        self.working_dir = f"{WebappCodeGenerator.webapp_working_dir}/{webapp_name}"
        self.frontend_working_dir = f"{self.working_dir}/public/{self.webapp_name}"
        self.design_spec_edits = self.design_spec_to_prompts(design_spec_file)

    def design_spec_to_prompts(self, design_spec_file_path: str) -> dict:
        prompts_to_file = {
            # filename: {inserts: [prompts], edits: [prompts]}
        }

        with open(design_spec_file_path, "r") as file:
            lines = file.readlines()
            filename = ""
            for line in lines:
                cleaned = line.strip().replace("\n", "")
                if cleaned:
                    if "#" in line:
                        if "# skip" not in line:
                            filename = line.replace("#", "")
                            filename = filename.strip()
                            if not filename in prompts_to_file:
                                prompts_to_file[filename] = {}
                                prompts_to_file[filename][self.create_key] = []
                                prompts_to_file[filename][self.edits_key] = []
                        continue
                    if "Create" in cleaned:
                        prompts_to_file[filename][self.create_key].append(self.to_insert_promt(cleaned.replace("Create: ", "")))
                    elif "Edit:" in cleaned:
                        prompts_to_file[filename][self.edits_key].append(self.to_edit_prompt(cleaned.replace("Edit: ", "")))

        return prompts_to_file

    def to_insert_promt(self, prompt):
        always_append = "In react write"
        return f"{always_append} {prompt}"

    def to_edit_prompt(self, prompt):
        expand = "Add react component"
        return f"{expand} {prompt}"

    def add_all_imports(self, creation_files):
        import_lines = []
        for file in creation_files:
            component_names = []
            with open(file, "r") as src_file:
                for line in src_file.readlines():
                    component = self.get_exported_component(line)
                    if component:
                        component_names.append(component.replace("\n", ""))
            if component_names:
                module_name = file.split("/")[-1].replace(".js", "")
                import_lines.append("import {" + ", ".join(component_names) + "} from './" + module_name +"'")
        return import_lines

    def get_exported_component(self, line):
        export_variants = [
            "export const ",
            "export default ",
            "export function ",
            "export "
        ]

        for pattern in export_variants:
            if pattern in line:
                return line.replace(pattern, "").replace(";", "").split(" ")[0]

    def do_inserts(self):
        for file_name in self.design_spec_edits:
            absolute_path = f"{self.frontend_working_dir}/{file_name}"
            file_lines = []
            with open(absolute_path, "r") as opened_file:
                file_lines = opened_file.readlines()
            current_line = file_lines[0]
            import_indexes = 0

            while "import" in current_line:
                import_indexes += 1
                current_line = file_lines[import_indexes]

            for prompt in self.design_spec_edits[file_name][self.create_key]:
                response = self.creation_ai_interface.prompt(prompt)[0]
                response = self.remove_exports(self.remove_imports(response))
                import_indexes += 1
                file_lines.insert(import_indexes, response)
                import_indexes += 1
                file_lines.insert(import_indexes, "\n")
                import_indexes += 1
                file_lines.insert(import_indexes, "\n")

            with open(f"{absolute_path}", "w") as opened_file:
                opened_file.writelines(file_lines)

            sleep(2)

    def do_edits(self):
        for file_name in self.design_spec_edits:
            absolute_path = f"{self.frontend_working_dir}/{file_name}"
            new_lines_without_import = ""
            import_lines_string = ""

            with open(absolute_path, "r") as opened_file:
                for line in opened_file.readlines():
                    if "import" in line:
                        import_lines_string += line
                    else:
                        new_lines_without_import += line

            for prompt in self.design_spec_edits[file_name][self.edits_key]:
                # Model has a tendency to make imports up
                new_lines_without_import = self.edit_ai_interface.edit(new_lines_without_import, prompt)[0]
                new_lines_without_import = self.remove_exports(self.remove_imports(new_lines_without_import))

            # import_lines = import_cleanup(import_lines_string)
            new_lines = import_lines_string + self.remove_imports(new_lines_without_import)

            with open(f"{absolute_path}", "w") as opened_file:
                opened_file.write(new_lines)

            sleep(2)

    @staticmethod
    def remove_imports(code_section):
        split_code = code_section.split(";")
        split_code = [line for line in split_code if "import" not in line]
        return ";".join(split_code)

    @staticmethod
    def remove_exports(code_section):
        split_code = code_section.split(";")
        split_code = [line for line in split_code if "export" not in line]
        return ";".join(split_code)

    def init_webapp(self):
        try:
            subprocess.check_output(f"{BIN_DIR}/init_webapp.sh {self.working_dir}", shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)

    def spawn_frontend_debug_session(self):
        try:
            subprocess.check_output(f"{BIN_DIR}/debug_screen.sh {self.frontend_working_dir} frontend", shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)

def main(argv):
    code_generator = WebappCodeGenerator("test", f"{CONTENT_GENERATION_CONFIG}/test_spec.txt")
    print("Initializing webapp...")
    code_generator.init_webapp()
    code_generator.spawn_frontend_debug_session()
    print("Started frontend debug session...")
    sleep(3)
    code_generator.do_inserts()
    print("Inserted test...")
    code_generator.do_edits()
    print("updated dom...")


if __name__ == "__main__":
    exit(main(sys.argv))
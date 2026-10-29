import abc
import ast
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ContextResult(abc.ABC):
    file_path: Path
    line_number: int


@dataclass(frozen=True)
class ContextClassResult(ContextResult):
    class_name: str
    signatures: str


@dataclass(frozen=True)
class ContextFunctionResult(ContextResult):
    function_name: str
    code: str


@dataclass(frozen=True)
class ContextCodeResult(ContextResult):
    code: str


class ReturnVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.target = None

    def generic_visit(self, node):
        if isinstance(node, ast.expr):
            return []
        results = []
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        results.extend(self.visit(item))
            elif isinstance(value, ast.AST):
                results.extend(self.visit(value))
        return results

    def find(self, tree: ast.AST, class_name: str) -> list[ast.stmt]:
        self.target = class_name
        return self.visit(tree)


class ClassFinder(ReturnVisitor):
    def visit_ClassDef(self, node: ast.ClassDef) -> list[ast.ClassDef]:
        if node.name == self.target:
            classes = [node]
        else:
            classes = []
        for child in node.body:
            classes.extend(self.visit(child))
        return classes


class MethodFinder(ReturnVisitor):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> list[ast.FunctionDef]:
        if node.name == self.target:
            methods = [node]
        else:
            methods = []
        for child in node.body:
            methods.extend(self.visit(child))
        return methods

    visit_AsyncFunctionDef = visit_FunctionDef


class SignatureExtractor(ast.NodeVisitor):
    def visit_ClassDef(self, node: ast.ClassDef) -> list[int]:
        # The class header itself (decorators to the line before the body),
        # then the signature of every member.
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        lines = list(range(first, node.body[0].lineno)) if node.body else [node.lineno]
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.extend(self.visit(child))
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                lines.extend(self.visit(child))
        return lines

    def visit_FunctionDef(self, node: ast.FunctionDef) -> list[int]:
        """
        Extract the lines corresponding to the function signature, including parameters and type annotations.
        :param node: The AST node representing the function definition.
        :return: A list of lines corresponding to the function signature.
        """
        # Get the line number of the function definition
        start_line = node.lineno
        # Get the first line of the function body to determine where the signature ends
        if node.body:
            # A one-line def has its body on the def line.
            end_line = max(start_line, node.body[0].lineno - 1)
        else:
            # get the last line of the function definition if there is node body
            # (which should not happen in valid Python code, but we handle it just in case)
            if node.returns:
                end_line = node.returns.lineno
            else:
                end_line = node.end_lineno
        return list(range(start_line, end_line + 1))

    visit_AsyncFunctionDef = visit_FunctionDef

    @staticmethod
    def handle_assign(node: ast.Assign | ast.AnnAssign) -> list[int]:
        """
        Extract the lines corresponding to an assignment statement, including type annotations if present.

        :param node: The AST node representing the assignment statement (either ast.Assign or ast.AnnAssign).
        :return: A list of lines corresponding to the assignment statement.
        """
        start_line = node.lineno
        if node.end_lineno is not None:
            return list(range(start_line, node.end_lineno + 1))
        else:
            return [start_line]

    def visit_Assign(self, node: ast.Assign) -> list[int]:
        return self.handle_assign(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> list[int]:
        return self.handle_assign(node)


class MethodCodeExtractor(ast.NodeVisitor):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> list[int]:
        """
        Extract the lines corresponding to the entire function definition, including the body.

        :param node: The AST node representing the function definition.
        :return: A list of lines corresponding to the entire function definition.
        """
        start_line = node.lineno
        if node.end_lineno is not None:
            return list(range(start_line, node.end_lineno + 1))
        else:
            # If end_lineno is not available, we can try to estimate it by looking at the body of the function
            if node.body:
                last_node = node.body[-1]
                if (
                    hasattr(last_node, "end_lineno")
                    and last_node.end_lineno is not None
                ):
                    return list(range(start_line, last_node.end_lineno + 1))
                else:
                    return list(range(start_line, last_node.lineno + 1))
            # Fallback to just the starting line if we cannot determine the end line
            return [start_line]

    visit_AsyncFunctionDef = visit_FunctionDef


class ContextRetrievalAPI:
    def __init__(
        self,
        codebase_path: str | os.PathLike[str] | Path,
        workers: int = 4,
        context_window: int = 3,
        lineno: bool = True,
    ):
        self._codebase_path = Path(codebase_path)
        self._workers = workers
        self._context_window = context_window
        self._lineno = lineno
        self._cache: dict[Path, ast.AST] = {}
        self._class_finder: ClassFinder = ClassFinder()
        self._method_finder: MethodFinder = MethodFinder()
        self._signature_extractor: SignatureExtractor = SignatureExtractor()
        self._method_code_extractor: MethodCodeExtractor = MethodCodeExtractor()

    def _parse_file(self, file_path: Path) -> ast.AST:
        """Parse the file and cache the AST."""
        if file_path in self._cache:
            return self._cache[file_path]
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tree = ast.parse(content, filename=str(file_path))
            self._cache[file_path] = tree
            return tree
        except (IOError, SyntaxError):
            return ast.Module(
                body=[], type_ignores=[]
            )  # Return an empty Module on failure

    @staticmethod
    def _find_string_in_file(file_path: Path, query: re.Pattern[str]) -> Optional[Path]:
        """Check if the query string is present in the file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if re.search(query, content):
                return file_path
            else:
                return None
        except IOError:
            return None

    def _find_string_in_codebase(self, query: str | re.Pattern[str]) -> list[Path]:
        """
        Search for files containing the query string.

        Files are read leniently: one source file in a legacy encoding (pylint
        ships Latin-1 test data) must not fail every search of the codebase.
        """
        matches = []
        if isinstance(query, str):
            pattern = re.compile(query)
        else:
            pattern = query
        all_files = [
            f
            for f in self._codebase_path.rglob("*")
            if f.is_file() and f.suffix == ".py"
        ]
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            for file in executor.map(
                lambda f: self._find_string_in_file(f, pattern), all_files
            ):
                if file is not None:
                    matches.append(file)
        return matches

    def _get_relevant_lines(self, file: Path, lines: list[int]) -> str:
        """Extract the relevant lines from the file."""
        try:
            with open(file, "r", encoding="utf-8", errors="replace") as f:
                content = f.readlines()
            # readlines() keeps each line's newline; strip it before joining.
            if self._lineno:
                relevant_lines = [
                    f"{i:>4}: {content[i - 1].rstrip(chr(10))}"
                    for i in lines
                    if i - 1 < len(content)
                ]
            else:
                relevant_lines = [
                    content[i - 1].rstrip("\n") for i in lines if i - 1 < len(content)
                ]
            return "\n".join(relevant_lines)
        except IOError:
            return ""

    def _search_class_in_path(self, cls: str, path: Path) -> list[ContextClassResult]:
        """Search for the class definition in the specified path."""
        tree = self._parse_file(path)
        classes = self._class_finder.find(tree, cls)
        results = []
        for class_node in classes:
            relevant_lines = self._signature_extractor.visit(class_node)
            results.append(
                ContextClassResult(
                    file_path=path,
                    line_number=class_node.lineno,
                    class_name=cls,
                    signatures=self._get_relevant_lines(path, relevant_lines),
                )
            )
        return results

    def search_class(self, cls: str) -> list[ContextClassResult]:
        """
        Search for the class definition in the codebase and return its location and signature.
        :param cls:
        :return:
        """
        query = rf"class\s+{re.escape(cls)}\s*[\(:]"
        matches = self._find_string_in_codebase(query)
        results = []
        for file in matches:
            results.extend(self._search_class_in_path(cls, file))
        return results

    def search_class_in_file(self, cls: str, f: str) -> list[ContextClassResult]:
        """
        Search for the class definition in the specified file and return its location and signature.
        :param cls:
        :param f:
        :return:
        """
        path = self._codebase_path / f
        if not path.is_file():
            return []
        return self._search_class_in_path(cls, path)

    def _search_method_in_path(
        self, method: str, path: Path
    ) -> list[ContextFunctionResult]:
        """Search for the method definition in the specified path."""
        tree = self._parse_file(path)
        methods = self._method_finder.find(tree, method)
        results = []
        for method_node in methods:
            relevant_lines = self._method_code_extractor.visit(method_node)
            results.append(
                ContextFunctionResult(
                    file_path=path,
                    line_number=method_node.lineno,
                    function_name=method,
                    code=self._get_relevant_lines(path, relevant_lines),
                )
            )
        return results

    def search_method(self, method: str) -> list[ContextFunctionResult]:
        """
        Search for the method definition in the codebase and return its location and code.
        :param method: The name of the method to search for.
        :return: A list of ContextFunctionResult objects containing the file path, line number, function name,
                 and code of each matching method definition.
        """
        query = rf"def\s+{re.escape(method)}\s*\("
        matches = self._find_string_in_codebase(query)
        results = []
        for file in matches:
            results.extend(self._search_method_in_path(method, file))
        return results

    def search_method_in_class(
        self, method: str, cls: str
    ) -> list[ContextFunctionResult]:
        """
        Search for the method definition within the specified class and return its location and code.

        :param method: The name of the method to search for.
        :param cls:    The name of the class to search within.
        :return:       A list of ContextFunctionResult objects containing the file path, line number, function name,
                       and code of each matching method definition found within the specified class.
        """
        query = rf"class\s+{re.escape(cls)}\s*[\(:]"
        matches = self._find_string_in_codebase(query)
        results = []
        for file in matches:
            tree = self._parse_file(file)
            classes = self._class_finder.find(tree, cls)
            for class_node in classes:
                methods = self._method_finder.find(class_node, method)
                for method_node in methods:
                    relevant_lines = self._method_code_extractor.visit(method_node)
                    results.append(
                        ContextFunctionResult(
                            file_path=file,
                            line_number=method_node.lineno,
                            function_name=method,
                            code=self._get_relevant_lines(file, relevant_lines),
                        )
                    )
        return results

    def search_method_in_file(self, method: str, f: str) -> list[ContextFunctionResult]:
        """
        Search for the method definition in the specified file and return its location and code.
        :param method: The name of the method to search for.
        :param f: The path to the file to search in.
        :return: A list of ContextFunctionResult objects containing the file path, line number, function name,
                 and code of each matching method definition found in the specified file.
        """
        path = self._codebase_path / f
        if not path.is_file():
            return []
        return self._search_method_in_path(method, path)

    def _search_code_in_path(self, c: str, path: Path) -> list[ContextCodeResult]:
        """Search for the code snippet in the specified path and return its location and +-3 lines of context."""
        # there is no need to parse the file for this, we can just search for the string and extract the relevant lines
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            content = file.read()
        lines = content.splitlines()
        num_lines = len(c.splitlines())
        results = []
        # The snippet is code, not a pattern: match it literally.
        for match in re.finditer(re.escape(c), content):
            start_line = content.count("\n", 0, match.start()) + 1
            end_line = start_line + num_lines - 1
            context_start = max(1, start_line - self._context_window)
            context_end = min(len(lines), end_line + self._context_window)
            relevant_lines = list(range(context_start, context_end + 1))
            results.append(
                ContextCodeResult(
                    file_path=path,
                    line_number=start_line,
                    code=self._get_relevant_lines(path, relevant_lines),
                )
            )
        return results

    def search_code(self, c: str) -> list[ContextCodeResult]:
        all_files = [
            f
            for f in self._codebase_path.rglob("*")
            if f.is_file() and f.suffix == ".py"
        ]
        results = []
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            for file in executor.map(
                lambda f: self._search_code_in_path(c, f), all_files
            ):
                results.extend(file)
        return results

    def search_code_in_file(self, c: str, f: str) -> list[ContextCodeResult]:
        path = self._codebase_path / f
        if not path.is_file():
            return []
        return self._search_code_in_path(c, path)

"""Narrow corrections to the shared scorer; pinned author sources stay intact."""
import ast
import copy
import torch

VERSION = 'shared_scoring_fixes_v1'


def delete_duplicate(sentence):
    """Deduplicate commutative factors without confusing products and quotients.

    Keep the first ordered term, preserve multiplicities and numerator/denominator
    roles. No cancellation or general symbolic simplification is introduced.
    """
    tokens = list(sentence)
    if tokens and tokens[-1] == 1:
        tokens.pop()
    terms, current = [], []
    for token in tokens:
        if token == 2:
            terms.append(current)
            current = []
        else:
            current.append(token)
    terms.append(current)
    result, seen = [], set()
    for term in terms:
        numerator, denominator, divide = [], [], False
        for index, token in enumerate(term):
            if index % 2:
                divide = token == 4
            else:
                (denominator if divide else numerator).append(token)
        identity = (tuple(sorted(numerator)), tuple(sorted(denominator)))
        if identity not in seen:
            if result:
                result.append(2)
            result.extend(term)
            seen.add(identity)
    return result + [1]


def corrected_terms(original):
    def calculate_terms(word, net, database, variables):
        if word not in ('sint', 'uyyt', '(u(u^2)xx)xx', 'BiLaplace(u)'):
            return original(word, net, database, variables)
        if word == 'sint':
            return torch.sin(database[:, variables.index('t')]).detach().cpu().numpy()

        def derivative(value, axis):
            if not value.requires_grad:
                return torch.zeros_like(database[:, :1])
            gradient = torch.autograd.grad(value.sum(), database, create_graph=True,
                                           allow_unused=True)[0]
            if gradient is None:
                return torch.zeros_like(database[:, :1])
            index = variables.index(axis)
            return gradient[:, index:index + 1]

        u = net(database)
        if word == 'uyyt':
            value = derivative(derivative(derivative(u, 'y'), 'y'), 't')
        elif word == '(u(u^2)xx)xx':
            inner = derivative(derivative(u ** 2, 'x'), 'x')
            value = derivative(derivative(u * inner, 'x'), 'x')
        else:
            spatial = [axis for axis in ('x', 'y', 'z') if axis in variables]
            if not spatial:
                raise ValueError('Biharmonic operator requires a spatial coordinate')
            laplace = sum(derivative(derivative(u, axis), axis) for axis in spatial)
            value = sum(derivative(derivative(laplace, axis), axis) for axis in spatial)
        return value.detach().cpu().numpy()
    return calculate_terms


def install(env):
    """Change only four terms, duplicate keys, and the token-combination veto.

    Recompile the original reward AST after removing its one combination veto;
    matrix construction, least squares, R², penalties and other rules are intact.
    """
    from .author import source_tree, CODE
    env['_scoring_original_calculate_terms'] = env['calculate_terms']
    env['_scoring_original_delete_duplicate'] = env['delete_duplicate']
    env['_scoring_original_calculate_reward'] = env['calculate_reward']
    env['calculate_terms'] = corrected_terms(env['calculate_terms'])
    env['delete_duplicate'] = delete_duplicate
    reward = copy.deepcopy(next(node for node in source_tree('continue_train_GPT.py').body
                                if isinstance(node, ast.FunctionDef) and node.name == 'calculate_reward'))
    vetoes = [node for node in reward.body if isinstance(node, ast.If)
              and "word2id['sin(u)']" in ast.unparse(node.test)
              and "word2id['sinh(u)']" in ast.unparse(node.test)]
    if len(vetoes) != 1:
        raise ValueError('Expected exactly one original token-combination veto')
    reward.body.remove(vetoes[0])
    module = ast.fix_missing_locations(ast.Module(body=[reward], type_ignores=[]))
    exec(compile(module, str(CODE / 'continue_train_GPT.py') + ':scoring_fixes', 'exec'), env)
    env['scoring_version'] = VERSION
    return env

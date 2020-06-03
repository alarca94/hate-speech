class Node:

    def __init__(self, attribute=None, avail_attrs=[], depth=None, children=dict(), terminal_value=None):
        self.terminal_value = terminal_value
        self.attribute = attribute
        self.avail_attrs = avail_attrs
        self.children = children

        self.depth = depth

    def add_child(self, value, child):
        self.children[value] = child

    def add_child_binary(self, leftOrRight, values, child):
        self.children[leftOrRight] = [values, child]

    def set_terminal(self, terminal_value):
        self.terminal_value = terminal_value

    def get_instances(self):
        out = set()
        for val in self.children.values():
            if isinstance(val, Node):
                out.update(val.get_instances())
            else:
                out.update(set(val))

        return list(out)

    def get_instances_binary(self):
        out = set()
        for x, val in self.children.values():
            if isinstance(val, Node):
                out.update(val.get_instances())
            else:
                out.update(set(val))

        return list(out)

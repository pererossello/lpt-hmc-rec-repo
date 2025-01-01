import os
import json


class Grove:

    def __init__(self, grove_dir):
        self.grove_dir = os.path.abspath(grove_dir)
        self.forest_dir = os.path.dirname(self.grove_dir)
        self.grove_name = os.path.basename(self.grove_dir)

        self.handle_grove_dir()
        self._build_grove()

    def handle_grove_dir(self):
        if os.path.exists(self.grove_dir):
            print(
                f"Grove '{self.grove_name}' at forest \n{self.forest_dir} \nexists.\n"
            )
        else:
            print(
                f"Grove '{self.grove_name}' at forest \n{self.forest_dir}"
                "\ndoes NOT exist. Creating.\n"
            )
            os.makedirs(self.grove_dir, exist_ok=True)

    def _build_grove(self):
        def add_to_grove(parent_node: Node, path_parts, full_path):
            if not path_parts:
                return
            current_idx = int(path_parts[0][2:])  # CH00 → 00

            if current_idx not in parent_node.children:
                parent_node.add_child(
                    current_idx, os.path.join(full_path, path_parts[0])
                )

            add_to_grove(parent_node.children[current_idx], path_parts[1:], full_path)

        # Initialize soil (base) node
        self.grove = Node(
            -1, self.grove_dir
        )  # Soil node idx = -1 (acts as the base of the grove)

        for trunk, dirs, _ in os.walk(self.grove_dir):
            for d in dirs:
                if d.startswith("CH"):
                    relative_path = os.path.relpath(
                        os.path.join(trunk, d), self.grove_dir
                    )
                    path_parts = relative_path.split(os.sep)
                    add_to_grove(self.grove, path_parts, trunk)

        print(f"Grove initialized with {len(self.grove.children)} trunks.")


    """
    Methods to add members (trunks and branches)
    """

    def add_trunk(self):
        """Add a single trunk to the grove."""
        self.add_n_trunks(1)

    def add_n_trunks(self, n):
        """Add 'n' trunks to the grove."""
        n_trunks = len(self.grove.children)
        for _ in range(n):
            self._add_member([])
        self.save_grove_to_txt()

        print(f"Grove has {n_trunks} trunks. Adding {n} more trunks.")
        n_roots = len(self.grove.children)
        print(f"Grove has now {n_roots} trunks.")
        

    def add_branch(self, idx_list):
        """Add a branch or sub-branch at the specified path.
        Args:
            path_list (list): List of indices specifying the trunk/branch path.
        """
        if not idx_list:
            raise ValueError(
                "Cannot add a branch directly to the grove. Specify a trunk or branch path."
            )

        # Use the internal _add_member method to handle the branch creation
        self._add_member(idx_list)
        self.save_grove_to_txt()


    def _add_member(self, idx_list=[]):
        def get_node_at_path(node: Node, idx_list):
            current = node
            for idx in idx_list:
                if idx not in current.children:
                    raise ValueError(
                        f"Invalid path {idx_list}. Trunk or branch {idx} not found."
                    )
                current = current.children[idx]
            return current

        # If empty path_list, we add a trunk directly under the grove
        target_node = (
            self.grove if not idx_list else get_node_at_path(self.grove, idx_list)
        )

        # Determine the next available index for the new member
        next_idx = len(target_node.children)
        member_name = f"CH{next_idx:02d}"
        member_path = os.path.join(target_node.path, member_name)

        # Create the directory on disk
        os.makedirs(member_path, exist_ok=True)

        # Add the member (trunk/branch) to the grove
        target_node.add_child(next_idx, member_path)



    """
    Grove to text methods
    """
    def _get_grove_as_text(self, node=None, prefix="", is_last=True):
        if node is None:
            node = self.grove  # Start from the root node (TreeNode)

        text = ""
        keys = sorted(node.children.keys())  # Sort children by index
        for i, key in enumerate(keys):
            child = node.children[key]
            connector = "└── " if i == len(keys) - 1 else "├── "
            text += f"{prefix}{connector}CH{child.idx:02d}\n"
            extension = "    " if i == len(keys) - 1 else "│   "
            text += self._get_grove_as_text(child, prefix + extension, is_last=(i == len(keys) - 1))
        return text

    def print_grove(self):
        print("Grove Structure:\n")
        print(self._get_grove_as_text())

    def save_grove_to_txt(self):
        txt_path = os.path.join(self.grove_dir, "grove_structure.txt")
        with open(txt_path, "w") as f:
            f.write(self._get_grove_as_text())
        #print(f"Grove structure saved to {txt_path}")


class Node:
    def __init__(self, idx, path, parent=None):
        self.idx = idx  # Store as integer (e.g., 0 for CH00)
        self.path = path  # Full path to this node
        self.parent = parent  # Reference to parent node
        self.children = (
            {}
        )  # Children indexed by integers (e.g., {0: TreeNode, 1: TreeNode})
        self.level = parent.level + 1 if parent else 0

        self.hmc_config = None

    def add_child(self, idx, path):
        child = Node(idx, path, parent=self)
        self.children[idx] = child
        return child

    def __repr__(self):
        return f"TreeNode(idx={self.idx})"

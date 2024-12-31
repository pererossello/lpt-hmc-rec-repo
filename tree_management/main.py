import os
import json

class HMCTree:

    def __init__(self, ground_dir):
        self.ground_dir = os.path.abspath(ground_dir)
        self.gdir_parent = os.path.dirname(self.ground_dir)
        self.gdir_basename = os.path.basename(self.ground_dir)

        self._check_existing_tree_txt()

        self.handle_ground_dir()
        self._build_tree()
        self._classify_tree()
        self.save_tree_to_txt()

    def handle_ground_dir(self):
        if os.path.exists(self.ground_dir):
            print(f"Ground dir '{self.gdir_basename}' at \n{self.gdir_parent} \nexists\n")
        else:
            print(f"Ground dir '{self.gdir_basename}' at \n{self.gdir_parent}"
                "\ndoes NOT exist. Creating\n")
            os.makedirs(self.ground_dir, exist_ok=True)


    def _build_tree(self):
        def add_to_nested_tree(nested_tree, path_parts):
            if not path_parts:
                return
            current = int(path_parts[0][2:])  # Extract integer from 'CHXX'
            if current not in nested_tree:
                nested_tree[current] = {}
            add_to_nested_tree(nested_tree[current], path_parts[1:])

        # Initialize an empty nested tree
        self.tree = {}

        # Walk through the ground dir and build the nested tree
        for root, dirs, _ in os.walk(self.ground_dir):
            for d in dirs:
                if d.startswith("CH"):
                    # Get relative path components for nesting
                    relative_path = os.path.relpath(os.path.join(root, d), self.ground_dir)
                    path_parts = relative_path.split(os.sep)
                    add_to_nested_tree(self.tree, path_parts)

        print(f"Tree structure initialized with {len(self.tree)} root nodes.")


    def _create_root(self):
        existing_chains = sorted([d for d in os.listdir(self.ground_dir) if d.startswith("CH")])
        next_chain_number = len(existing_chains)
        root_name = f"CH{next_chain_number:02d}"
        root_path = os.path.join(self.ground_dir, root_name)
        
        os.makedirs(root_path, exist_ok=True)

    def create_root(self):
        self._create_root()
        self.refresh()

    def create_n_roots(self, n):
        for _ in range(n):
            self._create_root()

        print(f"Created {n} roots.")
        self.refresh()
        


    def create_n_branches_at_tips(self, n):
        if self.tree_type == "Irregular":
            print("Tree is irregular. Branching at tips is not allowed.")
            return
        
        def get_leaf_nodes(node, path, depth, max_depth, leaves):
            if not node:  # Reached a leaf
                if depth == max_depth:
                    leaves.append(path[:])  # Copy of path to avoid mutation
                return
            
            for idx, child in node.items():
                path.append(idx)
                get_leaf_nodes(child, path, depth + 1, max_depth, leaves)
                path.pop()

        def get_max_depth(node, depth=0):
            if not node:
                return depth
            return max(get_max_depth(child, depth + 1) for child in node.values())

        # Calculate max depth and find leaves
        max_depth = get_max_depth(self.tree)
        leaves = []
        get_leaf_nodes(self.tree, [], 0, max_depth, leaves)

        # Use _create_branch to add branches at tips
        for leaf_path in leaves:
            for _ in range(n):
                self._create_branch(leaf_path)
        
        
        print(f"Created {n} branches at each tip of the tree.")
        self.refresh()


    def _create_branch(self, path_list):
        def get_node_at_path(tree, path_list):
            current = tree
            for idx in path_list:
                if idx not in current:
                    raise ValueError(f"Path {path_list} is invalid. Node {idx} does not exist.")
                current = current[idx]
            return current
        
        if not self.tree or len(self.tree) == 0:
            raise ValueError("Cannot create branch. The tree has no roots.")
        
        # Navigate to the target node or raise an error
        try:
            target_node = get_node_at_path(self.tree, path_list)
        except ValueError as e:
            print(e)
            return

        # Create the next sub-branch
        next_branch_number = len(target_node)
        target_node[next_branch_number] = {}

        # Create the corresponding folder on disk
        branch_path = os.path.join(self.ground_dir, *self._path_from_list(path_list), f"CH{next_branch_number:02d}")
        os.makedirs(branch_path, exist_ok=True)

    def _path_from_list(self, path_list):
        path = []
        current = self.tree
        for idx in path_list:
            if idx in current:
                path.append(f"CH{idx:02d}")
                current = current[idx]
            else:
                break
        return path


    def _get_tree_as_text(self, node=None, prefix="", is_last=True):
        if node is None:
            node = self.tree  # Start from the root
        
        
        text = ""
        keys = sorted(node.keys())
        for i, key in enumerate(keys):
            connector = "└── " if i == len(keys) - 1 else "├── "
            text += f"{prefix}{connector}CH{key:02d}\n"
            extension = "    " if i == len(keys) - 1 else "│   "
            text += self._get_tree_as_text(node[key], prefix + extension, is_last=(i == len(keys) - 1))
        return text


    def print_tree(self):
        title = f"{self.tree_type} Tree\n\n"
        tree_text = self._get_tree_as_text()
        print(title)
        print(tree_text if tree_text else "Tree is empty.")

    def save_tree_to_txt(self):
        title = f"{self.tree_type} Tree\n\n"
        tree_text = self._get_tree_as_text()
        file_path = os.path.join(self.ground_dir, "tree_structure.txt")

        if not tree_text:
            with open(file_path, "w") as f:
                f.write('Empty Tree')
            return
        
        
        with open(file_path, "w") as f:
            f.write(title)
            f.write(tree_text)
        

    def _classify_tree(self):
        def get_depths(node, depth=0):
            if not node:
                return [depth]
            depths = []
            for child in node.values():
                depths.extend(get_depths(child, depth + 1))
            return depths

        def is_uniform(node):
            if not node:
                return True
            num_children = [len(child) for child in node.values()]
            if len(set(num_children)) > 1:
                return False
            return all(is_uniform(child) for child in node.values())

        if not self.tree:
            self.tree_type = "Empty Tree"
            return

        depths = get_depths(self.tree)
        min_depth = min(depths)
        max_depth = max(depths)

        if len(set(depths)) == 1 and is_uniform(self.tree):
            self.tree_type = "Regular"
        elif min_depth == max_depth:
            self.tree_type = "Subregular"
        else:
            self.tree_type = "Irregular"

        print(f"Tree classified as: {self.tree_type}")

    def refresh(self):
        self._build_tree()
        self._classify_tree()
        print("Tree structure refreshed and reclassified.")
        
        # Resave if the tree was previously saved
        if self.tree_txt_exists:
            self.save_tree_to_txt()

    def _check_existing_tree_txt(self):
        txt_path = os.path.join(self.ground_dir, "tree_structure.txt")
        if os.path.exists(txt_path):
            self.tree_txt_exists = True
            self.tree_txt_path = txt_path
        else:
            self.tree_txt_path = None
            self.tree_txt_exists = False


class TreeNode:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.children = []
        self.parent = None



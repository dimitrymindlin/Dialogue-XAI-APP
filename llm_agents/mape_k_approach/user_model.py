class UnderstandingItem:
    def __init__(self, label, description):
        """
        Each understanding item has a label (a short, meaningful identifier)
        and a description (a detailed explanation of the concept).
        """
        self.label = label
        self.description = description

    def __repr__(self):
        return f"{self.label}: {self.description}"


class UserModel:
    def __init__(self):
        self.understood = []
        self.misunderstood = []
        self.not_explained_yet = []
        self.shown_explanation = []

    def add_not_explained_yet(self, label, description):
        """Add an item to the not explained yet list."""
        item = UnderstandingItem(label, description)
        self.not_explained_yet.append(item)

    def mark_as_shown(self, label):
        """Move an item from 'not explained yet' to 'shown explanation'."""
        item = self._move_item(label, self.not_explained_yet)
        if item and item not in self.shown_explanation:
            self.shown_explanation.append(item)

    def mark_as_understood(self, label):
        """Move an item identified by its label to 'understood'."""
        item = self._move_item(label, self.not_explained_yet, self.shown_explanation)
        if item and item not in self.understood:
            self.understood.append(item)

    def mark_as_misunderstood(self, label):
        """Move an item identified by its label to 'misunderstood'."""
        item = self._move_item(label, self.not_explained_yet, self.shown_explanation)
        if item and item not in self.misunderstood:
            self.misunderstood.append(item)

    def _move_item(self, label, *sources):
        """Utility function to remove an item from sources and return it."""
        for source in sources:
            for item in source:
                if item.label == label:
                    source.remove(item)
                    return item
        return None

    def get_summary(self):
        """Return a summary of all understood, misunderstood, shown explanation, and not explained yet items."""
        return {
            "understood": [item.label for item in self.understood],
            "misunderstood": [item.label for item in self.misunderstood],
            "shown explanation": [item.label for item in self.shown_explanation],
            "not explained yet": [item.label for item in self.not_explained_yet]
        }

    def is_understood(self, label):
        """Check if an item identified by its label is marked as understood."""
        return any(item.label == label for item in self.understood)

    def is_misunderstood(self, label):
        """Check if an item identified by its label is marked as misunderstood."""
        return any(item.label == label for item in self.misunderstood)

    def is_shown_explanation(self, label):
        """Check if an item identified by its label is marked as shown explanation."""
        return any(item.label == label for item in self.shown_explanation)

    def is_not_explained_yet(self, label):
        """Check if an item identified by its label is marked as not explained yet."""
        return any(item.label == label for item in self.not_explained_yet)
class MyClass:
    class_variable = 10

    @classmethod
    def change_class_variable(cls, new_value=0):
        cls.class_variable += 1
from prototxt_parser.prototxt import parse

# Đọc nội dung của file .prototxt
with open('colorization_deploy_v2.prototxt', 'r') as file:
    input_string = file.read()

# Chuyển đổi nội dung thành dict Python
parsed_dict = parse(input_string)
print(parsed_dict)
```[_{{{CITATION{{{_1{prototxt-parser - PyPI](https://pypi.org/project/prototxt-parser/)


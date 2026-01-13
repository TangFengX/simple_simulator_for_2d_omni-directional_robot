import os
from param import *


def generate_param_header(file_path=HEAD_FILE_PATH, param_list=C_PARAM_LIST):
    """
    生成 C 语言参数头文件
    :param file_path: 输出文件路径 (例如 './param.h')
    :param param_list: 嵌套列表 [[宏名称, 宏内容, 注释], ...]
    """
    lines = []
    
    # 1. 写入 Header Guard
    lines.append("#ifndef __PARAM_H__")
    lines.append("#define __PARAM_H__\n")
    lines.append("//自动生成，请勿修改\n")
    
    # 2. 处理宏定义
    # 计算名称的最大长度，用于自动对齐
    if param_list:
        max_name_len = max(len(item[0]) for item in param_list)
    else:
        max_name_len = 20

    for item in param_list:
        name = item[0]
        value = str(item[1])
        comment = item[2] if len(item) > 2 and item[2] else ""
        
        # 格式化输出: #define NAME VALUE // COMMENT
        # 使用 ljust 实现名称对齐
        line = f"#define {name.ljust(max_name_len)} {value}"
        if comment:
            line += f" // {comment}"
        
        lines.append(line)
    
    # 3. 写入结尾
    lines.append("\n#endif // !__PARAM_H__")
    
    # 4. 执行写入
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"成功生成头文件: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"生成失败: {e}")



if __name__=="__main__":
    
    generate_param_header()
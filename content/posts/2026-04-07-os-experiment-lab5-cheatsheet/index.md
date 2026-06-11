+++
date = '2026-04-07T14:43:01+08:00'
draft = false
title = '操作系统实践——知识点与小demo总结（简单文件系统的实现）'
ShowToc = false
math = false
categories = ["cheatsheet"]
tags = [
    "OS", 
    "linux", 
    "File System"
]
+++

## 实验5：简单文件系统的实现

### 基础架构与持久化层

#### 宏定义

```c
#define IMAGE_FILE "fs.img"     // 磁盘镜像文件名
#define BLOCK_SIZE 512          // 每个数据块字节数（512字节，传统磁盘扇区大小）
#define DATA_BLOCKS 1024        // 数据块数量（1024个数据块，数据区512KB）
#define MAX_FILES 256           // 文件/目录数上限（DirEntry数组大小）
#define MAX_OPEN_FILES 32       // 同时打开文件数上限（OFT表大小）
#define MAX_NAME_LEN 28         // 文件名长度上限

#define FAT_FREE (-1)   // FAT表空闲块标记
#define FAT_END  (-2)   // FAT表链尾标记

#define FS_MAGIC 0x4D465331u /* "MFS1" */
```

- `FS_MAGIC` 用于识别文件系统格式，加载时校验合法性，防止用其他文件创建的文件误当作文件系统镜像。
  
  这里用一串十六进制数表示，可以解析为"MFS1"。

- `BLOCK_SIZE`与`DATA_BLOCKS` 定义的数据区大小（512KB）是**纯数据区大小**，不包括FAT表和目录项占用的元数据空间。

#### 结构体

##### `SuperBlock`（超级块）

```c
// 超级块
typedef struct {
    uint32_t magic;             // 魔法数字，用于识别文件系统镜像
    int32_t block_size;         // 单块字节数
    int32_t block_count;        // 总数据块数
    int32_t max_files;          // 最大文件/目录数
    int32_t root_dir_index;     // 根目录在entries数组中的索引
} SuperBlock;
```

- 这是文件系统的 **“身份证”**，存储在**镜像文件的最开头**。

- 在加载时，会首先读取并**校验**`magic`，如果不匹配则拒绝挂载，防止用普通文本文件或损坏数据误当文件系统加载。 具体见`load_filesystem()`的实现。

##### `DirEntry`（目录项）

```c
// 目录项
#pragma pack(push, 1)
typedef struct {
    char name[MAX_NAME_LEN];    // 28字节，文件名（固定长度）
    uint8_t type;               // 1字节，0=文件，1=目录（见EntryType）
    int32_t start_block;        // 4字节，FAT链起始块号（-1表示空文件）
    int32_t size;               // 4字节，文件字节数 或 目录子项数量
    int32_t parent;             // 4字节，父目录在entries中的索引
    uint8_t used;               // 1字节，0=空闲，1=占用
} DirEntry;                     // 总计42字节
#pragma pack(pop)
```

- 这里定义了目录项，即我们说的**文件控制块FCB**

- `#pragma pack(push, 1)`：这是**编译器指令**，强制编译器**不进行字节对齐填充**。
  
  默认情况下，结构体会按照4或8字节对齐，这会导致内存中插入“空洞”。
  
  这样设置的**紧凑布局**确保磁盘镜像大小精确可控，无浪费。

- `name` 固定28字节，注意这里**与C字符串不同**，不会以 `\0` 结尾，可以用`strncpy`写入，但需要小心越界。

- `parent` 是构建目录树的核心。根目录的`parent`会指向自身，形成递归终止条件。

##### `FsImage`（内存镜像）

```c
// 内存镜像
typedef struct {
    SuperBlock super;                           // 超级块
    int32_t fat[DATA_BLOCKS];                   // FAT表
    DirEntry entries[MAX_FILES];                // 目录项数组
    uint8_t data[DATA_BLOCKS * BLOCK_SIZE];     // 数据区
    int32_t saved_current_dir;                  // 退出时保存的当前目录索引
} FsImage;
```

- 这是一个**内存映射文件系统**的设计。
  
  整个镜像一次性 `fread` 进**内存**，操作完成后再 `fwrite` 回磁盘。

- `saved_current_dir` 会记录退出时的当前目录，下次启动自动恢复。

##### `OpenFile` （打开文件表项）

```c
// 打开文件表项
typedef struct {
    int used;           // 0=空闲，1=占用
    int entry_index;    // 对应 DirEntry 在 entries 数组中的索引
    int32_t offset;     // 当前读写偏移量（字节）
} OpenFile;
```

- 这是 **OFT 表的表项**，与 DirEntry 分离，专门跟踪**文件打开状态**。

- `offset` 跟踪读写位置，是**不同写模式**实现的关键。

##### `FileSystem`（运行时上下文）

```c
// 运行时上下文
typedef struct {
    FsImage image;      // 实际数据载体
    int mounted;        // 0=未挂载，1=已挂载
    int current_dir;    // 当前工作目录在 entries 中的索引
    OpenFile oft[MAX_OPEN_FILES];   // 打开文件表（OFT）
} FileSystem;
```

- 与 `FsImage` 不同，`FileSystem` 属于**会话层**（运行时状态，包括未保存到磁盘的当前目录和打开文件）；而 `FsImage` 是**持久化层**（存磁盘的样子）。

#### `load_filesystem()`加载文件系统

```c
static void load_filesystem(FileSystem *fs) {
    memset(fs, 0, sizeof(*fs));     // 防御性清零

    // 尝试加载镜像
    FILE *fp = fopen(IMAGE_FILE, "rb");
    if (!fp) {
        printf("No existing image found. Run my_format to initialize.\n");
        return;
    }

    // 校验（整读校验与魔数校验）
    size_t read = fread(&fs->image, 1, sizeof(fs->image), fp);
    fclose(fp);
    if (read != sizeof(fs->image) || fs->image.super.magic != FS_MAGIC) {
        printf("Invalid or corrupted image. Run my_format.\n");
        memset(&fs->image, 0, sizeof(fs->image));
        return;
    }

    // 恢复会话状态
    fs->mounted = 1;
    fs->current_dir = fs->image.saved_current_dir;
    if (fs->current_dir < 0 || fs->current_dir >= MAX_FILES ||
        !fs->image.entries[fs->current_dir].used ||
        fs->image.entries[fs->current_dir].type != ENTRY_DIR) {
        fs->current_dir = fs->image.super.root_dir_index;
    }

    // 初始化OFT
    reset_open_files(fs);
    printf("Loaded file system image from %s.\n", IMAGE_FILE);
}
```

- **调用时机**：在`main()`函数中的第一件事就是加载镜像：
  
  ```c
  int main(void) {
      FileSystem fs;
      load_filesystem(&fs);
      ……
  }
  ```

- 开始的**防御性清零**是系统编程的惯用手法。
  
  `FileSystem` 初始化时，内存中是随机值。若不清零，结构体中的标志位可能是**非0的野值**，导致后续逻辑混乱。

- `fopen(IMAGE_FILE, "rb")` 打开镜像文件
  
  - `"rb"` 是**二进制只读模式**
  
  - 如果**无镜像文件**（`fp=0`，首次运行或镜像被删除），**提示用户**运行 `my_format` 命令初始化。
  
  - 此时 `fs->mounted=0`（经过防御性清零），表示文件系统未挂载，符合逻辑。

- 加载镜像后，需要进行多方面的校验：
  
  - **整读校验**：检查镜像文件读出的内容大小与设计的结构体大小相同，即镜像中所有内容都被正常读出。
    
    其中读数据用到了 `fread()`，其函数原型为：
    
    ```c
    size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
    ```
    
    - `void *ptr`：内存缓冲区起始地址
    
    - `size_t size`：**每个数据单元的字节数**（这里取1表示按字节读取）
    
    - `size_t nmemb`：**要读取的单元数量**
    
    - `FILE *stream`：文件指针
    
    - 返回值：成功读取的单元数量，因为`size=1`，因此返回值等于实际读取到**字节数**
  
  - **魔数校验**：确保打开的文件是**本程序创建的镜像文件**
    
    假设其他文件被当成镜像文件打开，这里检查 `magic` 值无法正常加载；
    
    相反，镜像文件加载后，`magic` 值可以与宏定义的值匹配（这是在初始化镜像文件时做的工作，详见函数 `cmd_format()`）。
  
  - 如果未通过检验，提示镜像损坏，**清零image**，保持`fs->mounted=0`。

- 在恢复工作目录时，有详细的检查，**如果不符合打开条件，把根目录作为当前工作目录**：
  
  ```c
      fs->current_dir = fs->image.saved_current_dir;
      if (fs->current_dir < 0 ||  // 当前目录序列号<0
          fs->current_dir >= MAX_FILES ||    // 当前目录序列号>最大序列号
          !fs->image.entries[fs->current_dir].used ||    // 当前目录为空
          fs->image.entries[fs->current_dir].type != ENTRY_DIR) {    // 序列号对应目录项不是目录
          fs->current_dir = fs->image.super.root_dir_index;
      }
  ```

#### `save_filesystem()`持久化镜像

```c
// 持久化镜像
static int save_filesystem(FileSystem *fs) {
    // 检查挂载状态
    if (!fs->mounted) {
        printf("File system not formatted; nothing to save.\n");
        return -1;
    }

    // 保存会话状态
    fs->image.saved_current_dir = fs->current_dir;

    // 整写镜像
    FILE *fp = fopen(IMAGE_FILE, "wb");
    if (!fp) {
        perror("Failed to open image for writing");
        return -1;
    }

    // 完整性校验
    size_t written = fwrite(&fs->image, 1, sizeof(fs->image), fp);
    fclose(fp);
    if (written != sizeof(fs->image)) {
        printf("Failed to write complete image.\n");
        return -1;
    }
    return 0;
}
```

- **调用时机**：用户使用 `my_exitsys`时：
  
  ```c
  static void cmd_exitsys(FileSystem *fs, int *running) {
      if (!check_mounted(fs)) {
          return;
      }
      if (save_filesystem(fs) == 0) {
          printf("File system saved. Bye!\n");
          *running = 0;
      }
  }
  ```

- 之所以要检查挂载状态 `fs->mounted`，是因为我们要防止生成空的或者损坏的镜像文件

- 写镜像时，用`fopen(IMAGE_FILE, "wb")`，`"wb"`表示**二进制写（覆盖模式）**：
  
  - 若 `fs.img` 文件已存在，直接清空重写；
  
  - 若不存在，则创建 `fs.img` 文件。
  
  写入时，整个内存内容一次性写入，与`fread`一次性读对称。

#### `reset_open_files()`重置打开文件表（OFT）

```c
static void reset_open_files(FileSystem *fs) {
    for (int i = 0; i < MAX_OPEN_FILES; ++i) {
        fs->oft[i].used = 0;
        fs->oft[i].entry_index = -1;
        fs->oft[i].offset = 0;
    }
}
```

- 该函数会**初始化进程级的文件描述符表（OFT，Open File Table）**，将OFT所有表项标记为“未使用”状态。
  
  - `used`：设置为0，即空闲，后续可以供 `cmd_open()` 分配。
  
  - `entry_index`：设置为 **-1**，表示**未关联任何目录项**。
  
  - `offset`：设置为0，将读写位置重置到文件开头位置。

- **调用时机**
  
  - 系统加载时（`load_filesystem()`）
    
    ```c
    static void load_filesystem(FileSystem *fs) {
        ……
        // 初始化OFT
        reset_open_files(fs);
        printf("Loaded file system image from %s.\n", IMAGE_FILE);
    }
    ```
  
  - 格式化时（`cmd_format()`）
    
    ```c
    static void cmd_format(FileSystem *fs) {
        ……
        reset_open_files(fs);
        printf("File system formatted.\n");
    }
    ```

#### `check_mounted()`检查挂载状态

```c
static int check_mounted(const FileSystem *fs) {
    if (!fs->mounted) {
        printf("File system is not formatted. Run my_format first.\n");
        return 0;
    }
    return 1;
}
```

- 封装的检查`fs->mounted`方法，注意**检查的是内存，而不是磁盘**。

- **调用时机**：所有的 `cmd_XXX()` 函数，用户的所有操作都需要先检查文件系统的挂载状态。

#### 相关问题

1. **请问文件系统如何处理 `FsImage` 的序列化与反序列化？**
   
   当前程序使用的是C结构体内存直接映射。
   
   序列化，即数据**从内存到磁盘**的过程，我们把整个`FsImage`结构体作为连续的字节流，原封不动地写入 `fs.img` 文件：
   
   ```c
   fwrite(&fs->image, 1, sizeof(fs->image), fp);
   ```
   
   相应的，反序列化，即数据**从磁盘到内存**的过程，也是将磁盘文件的二进制字节流，直接覆盖到 `FsImage` 结构体的内存地址。
   
   为了保证直接映射正常工作，我们做了以下工作：
   
   - 紧凑布局：我们使用了 `#pragma pack(push, 1)` ，强制结构体1字节对齐，避免空隙影响结构体数据读取。
   
   - 校验：采用`magic`，保证加载的文件一定是我们程序创建的镜像文件
   
   - 定长数组：结构体中的所有成员均为定长数组，保证 `sizeof(fs->image)` 在编译期就确定。

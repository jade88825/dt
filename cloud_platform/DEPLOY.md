# 智慧交通云平台 - Render.com 部署指南

## 部署步骤

### 1. 创建GitHub仓库

在 `F:\大唐` 目录下初始化Git:

```bash
cd F:\大唐
git init
git add cloud_platform/
git commit -m "Smart Traffic Cloud Platform"
git branch -M main
```

### 2. 推送到GitHub

```bash
# 在GitHub创建新仓库: smart-traffic-platform
git remote add origin https://github.com/你的用户名/smart-traffic-platform.git
git push -u origin main
```

### 3. 在Render.com部署

1. 打开 https://render.com 注册账号
2. 点击 "New +" → "Web Service"
3. 连接你的GitHub仓库
4. 配置:
   - Name: smart-traffic-platform
   - Environment: Python 3
   - Build Command: `pip install -r cloud_platform/requirements-deploy.txt`
   - Start Command: `cd cloud_platform && gunicorn app:app --timeout 120 --workers 1`
   - Plan: Free
5. 点击 "Create Web Service"
6. 等待构建完成 (约5-10分钟)
7. 访问分配的URL: `https://smart-traffic-platform.onrender.com`

### 4. 注意事项

- 免费版15分钟无访问会休眠, 首次唤醒约30秒
- CPU推理速度: 约2-5秒/张图片
- 模型文件22MB已包含在仓库中
- 视频文件12MB已包含在仓库中

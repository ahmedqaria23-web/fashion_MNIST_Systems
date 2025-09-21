# 🧠 Fashion MNIST Classification System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

نظام ذكي لتصنيف ملابس Fashion MNIST باستخدام التعلم العميق وواجهة API حديثة

[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-GitHub%20Codespaces-blue?logo=github)](https://codespaces.new/your-username/fashion-mnist-classifier)

</div>

## 📖 نظرة عامة

هذا المشروع يقدم نظامًا متكاملًا لتصنيف صور الملابس من مجموعة بيانات Fashion MNIST باستخدام:

- **نماذج CNN متقدمة** بدقة تصل إلى 95%
- **واجهة API حديثة** باستخدام FastAPI
- **لوحة تحكم تفاعلية** باللغة العربية
- **نظام حماية** بمفاتيح API

## 🎯 الميزات الرئيسية

### 🤖 النموذج الذكي

- نموذج CNN متقدم مع Batch Normalization
- دقة تصنيف تصل إلى 95% على بيانات الاختبار
- معالجة مسبقة متطورة للصور
- زيادة البيانات (Data Augmentation)

### 🌐 واجهة API

- **FastAPI** سريع وفعّال
- توثيق API تلقائي (Swagger UI)
- حماية بمفاتيح API
- دعم الصور بأنواع متعددة
- استجابة سريعة (< 500ms)

### 🎨 لوحة التحكم

- واجهة عربية تفاعلية
- سحب وإفلات الصور
- عرض النتائج برسومات بيانية
- سجل التنبؤات المحفوظ
- تصميم متجاوب (Responsive)

## 📊 مجموعة البيانات

### Fashion MNIST - 10 فئات

| الفئة | الاسم  | الوصف          |
| ---------- | ----------- | ------------------- |
| 0          | T-shirt/top | تيشرت          |
| 1          | Trouser     | بنطال          |
| 2          | Pullover    | سترة            |
| 3          | Dress       | فستان          |
| 4          | Coat        | معطف            |
| 5          | Sandal      | صندل            |
| 6          | Shirt       | قميص            |
| 7          | Sneaker     | حذاء رياضي |
| 8          | Bag         | حقيبة          |
| 9          | Ankle boot  | حذاء كاحل   |

## 🚀 البدء السريع

### المتطلبات المسبقة

- Python 3.8+
- pip (مدير حزم Python)

### التثبيت

1. **استنساخ المشروع**

```bash
git clone https://github.com/your-username/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

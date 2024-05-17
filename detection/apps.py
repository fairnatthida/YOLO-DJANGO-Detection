from django.apps import AppConfig #ไฟล์นี้มันถูกสร้างขึ้นในเองตอน สร้าง project


class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

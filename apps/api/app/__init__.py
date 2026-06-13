"""CourseGPT web application package.

The existing LangChain/RAG code remains in apps.api.pipelines and apps.worker.
This package is the product layer: auth, database models, route handlers, and
service adapters that call the existing pipeline without rewriting it.
"""

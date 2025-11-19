import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownRendererProps {
  content: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <div className="prose prose-sm md:prose-base max-w-none prose-slate dark:prose-invert prose-headings:font-semibold prose-a:text-medical-600 hover:prose-a:text-medical-500">
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({node, ...props}) => (
            <div className="overflow-x-auto my-4 border rounded-lg">
              <table className="min-w-full divide-y divide-slate-200" {...props} />
            </div>
          ),
          thead: ({node, ...props}) => (
            <thead className="bg-slate-50" {...props} />
          ),
          th: ({node, ...props}) => (
            <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider" {...props} />
          ),
          td: ({node, ...props}) => (
            <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 border-t border-slate-100" {...props} />
          ),
          ul: ({node, ...props}) => (
            <ul className="list-disc pl-5 my-2 space-y-1" {...props} />
          ),
          ol: ({node, ...props}) => (
            <ol className="list-decimal pl-5 my-2 space-y-1" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
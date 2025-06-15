import React, { useState } from 'react';
import { config } from '../config/api';
import { Button } from '@/components/ui/button';
import { useLanguage } from '@/contexts/LanguageContext';

const Scanner = () => {
  const { t } = useLanguage();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(config.endpoints.analyzeImage, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">{t('scanner.title')}</h1>
      
      <div className="space-y-4">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer inline-block"
          >
            <div className="space-y-2">
              <div className="text-gray-600">
                {selectedFile ? selectedFile.name : 'Click to select an image'}
              </div>
              <Button variant="outline">
                {selectedFile ? 'Change Image' : 'Select Image'}
              </Button>
            </div>
          </label>
        </div>

        {selectedFile && (
          <div className="flex justify-center">
            <Button
              onClick={handleUpload}
              disabled={loading}
              className="w-full"
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </Button>
          </div>
        )}

        {error && (
          <div className="text-red-500 text-center">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-6 space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Detected Ornaments</h2>
              <ul className="list-disc list-inside">
                {result.detected_ornaments.map((ornament: string, index: number) => (
                  <li key={index}>{ornament}</li>
                ))}
              </ul>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Analysis</h2>
              <p>{result.response}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Scanner; 
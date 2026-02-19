"use client";
import React, { useState, useCallback, useEffect } from 'react';
import Image from 'next/image';
import { useDropzone } from 'react-dropzone';

export default function SportsClassifier() {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
  const wakeUpServer = async () => {
    try {
      // Just a simple ping to the base URL or a health route
      await fetch(process.env.NEXT_PUBLIC__BACKEND_URL + '/health');
      console.log("Backend is awake and ready!");
    } catch (e) {
      console.log("Server is still warming up...");
    }
  };

  wakeUpServer();
}, []);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    setResult(null);
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    multiple: false
  });

  const uploadImage = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null); // Clear previous results

    try {
      const response = await fetch(process.env.NEXT_PUBLIC__BACKEND_URL + '/classify_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: image }),
      });
      
      const data = await response.json();
      setResult(data);
      console.log("Classification result:", data);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Backend server not reached. Is Flask running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-50 py-12 px-4">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl p-8 border border-slate-200">
        <h1 className="text-2xl font-bold text-slate-800 mb-2 text-center">Sport Classifier AI</h1>
        <p className="text-slate-500 mb-8 text-center">Upload an action shot to identify the sport</p>

        {/* Drag & Drop Area */}
        <div 
          {...getRootProps()} 
          className={`relative border-2 border-dashed rounded-xl p-10 transition-all cursor-pointer text-center
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-slate-400'}`}
        >
          <input {...getInputProps()} />
          
          {image ? (
            <div className="relative group">
              {/* Container must be relative and have a height for 'fill' to work */}
              <div className="relative h-64 w-full">
                <Image 
                  src={image} 
                  alt="Preview" 
                  fill
                  style={{ objectFit: 'contain' }} // Maintains aspect ratio without cropping
                  className="rounded-lg shadow-sm" 
                />
              </div>
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity rounded-lg">
                <p className="text-white text-sm font-medium">Click or drag to replace</p>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-4xl">📸</div>
              <p className="text-slate-600 font-medium">Drag & drop your image here</p>
              <p className="text-xs text-slate-400">Supports JPG, PNG (Max 5MB)</p>
            </div>
          )}
        </div>

        <button
          onClick={uploadImage}
          disabled={!image || loading}
          className="w-full mt-6 bg-slate-900 text-white py-3 rounded-lg font-bold hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition-all shadow-lg active:scale-[0.98]"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </span>
          ) : "Classify Image"}
        </button>

        {/* Display Results */}
        {result && (
          <div className="mt-8 space-y-4 animate-in fade-in duration-500">
            {result.length > 0 ? (
              // Case 1: Face(s) found and classified
              result.map((item, index) => (
                <div key={index} className="p-4 bg-green-50 border border-green-100 rounded-lg shadow-sm">
                  <h3 className="text-xs uppercase tracking-wider text-green-600 font-bold mb-1">
                    {result.length > 1 ? `Person ${index + 1}` : "Top Prediction"}
                  </h3>
                  <div className="flex justify-between items-end">
                    <p className="text-2xl text-slate-900 font-bold capitalize">{item.class}</p>
                    <p className="text-sm text-green-700 font-semibold">{item.probability}% match</p>
                  </div>
                </div>
              ))
            ) : (
              // Case 2: Result array is empty (No eyes/face found by OpenCV)
              <div className="p-4 bg-amber-50 border border-amber-200 rounded-lg flex items-center gap-3">
                <span className="text-xl">⚠️</span>
                <div>
                  <p className="text-amber-800 font-bold">Could not recognize face</p>
                  <p className="text-amber-700 text-sm">Please ensure the person's eyes are clearly visible.</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
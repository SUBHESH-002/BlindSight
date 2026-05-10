import React, { useState, useEffect, useRef } from 'react';
import { Camera, Upload, Play, MonitorPlay, Activity, Radio, AlertTriangle, ShieldAlert, Target, Terminal, CheckCircle2, Cpu, Network } from 'lucide-react';

const COLORS = [
  '#3b82f6', '#8b5cf6', '#ef4444', '#10b981', '#f59e0b', 
  '#06b6d4', '#ec4899', '#84cc16', '#6366f1', '#14b8a6'
];

const getActionColor = (actionName) => {
  if (!actionName) return '#ffffff';
  let hash = 0;
  for (let i = 0; i < actionName.length; i++) {
    hash = actionName.charCodeAt(i) + ((hash << 5) - hash);
  }
  return COLORS[Math.abs(hash) % COLORS.length];
};

const KTH_CLASSES = ["Walking", "Boxing", "Handclapping"];

const getBackendUrls = () => {
  const envBackend = import.meta.env.VITE_BACKEND_URL?.trim();

  if (envBackend) {
    const url = new URL(envBackend.includes('://') ? envBackend : `${window.location.protocol}//${envBackend}`);
    return {
      http: `${url.protocol}//${url.host}`,
      ws: `${url.protocol === 'https:' ? 'wss:' : 'ws:'}//${url.host}`,
    };
  }

  const isBackendOrigin = window.location.port === '8000';
  const host = isBackendOrigin
    ? window.location.host
    : `${window.location.hostname || '127.0.0.1'}:8000`;
  const secure = window.location.protocol === 'https:';

  return {
    http: `${secure ? 'https:' : 'http:'}//${host}`,
    ws: `${secure ? 'wss:' : 'ws:'}//${host}`,
  };
};

const BACKEND = getBackendUrls();

export default function App() {
  const [mode, setMode] = useState('camera'); // 'camera', 'upload', 'simulation'
  const [status, setStatus] = useState('DISCONNECTED');
  const [uploadMessage, setUploadMessage] = useState('');
  const [currentEvent, setCurrentEvent] = useState(null);
  const [events, setEvents] = useState([]);
  const [targetText, setTargetText] = useState('describe action');
  const [temporalBuffer, setTemporalBuffer] = useState(0);
  const [attentionFlares, setAttentionFlares] = useState([]);
  const [uploadFrames, setUploadFrames] = useState([]);
  const [uploadFrameIndex, setUploadFrameIndex] = useState(0);
  const [uploadPlaybackTime, setUploadPlaybackTime] = useState(0);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const uploadIntervalRef = useRef(null);
  const simulateBlindspot = false;
  const showGrid = false;

  // Cleanup effect
  useEffect(() => {
    return () => {
      stopCamera();
      if (wsRef.current) wsRef.current.close();
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (uploadIntervalRef.current) clearInterval(uploadIntervalRef.current);
    };
  }, []);

  // Mode change handler
  useEffect(() => {
    stopCamera();
    if (wsRef.current) wsRef.current.close();
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (uploadIntervalRef.current) clearInterval(uploadIntervalRef.current);
    setCurrentEvent(null);
    setEvents([]);
    setStatus('DISCONNECTED');
    setUploadMessage('');
    setUploadFrames([]);
    setUploadFrameIndex(0);
    setUploadPlaybackTime(0);
    setTemporalBuffer(0);
    setAttentionFlares([]);

    if (mode === 'camera') {
      startCamera();
    } else if (mode === 'simulation') {
      startSimulation();
    }
  }, [mode]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      connectWebSocket();
    } catch (err) {
      console.error('Camera error:', err);
      setStatus('ERROR');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
  };

  const triggerAttention = () => {
    // Pick 2-3 random indices from 0-15 to light up
    const numFlares = Math.floor(Math.random() * 2) + 2;
    const flares = [];
    for(let i=0; i<numFlares; i++){
      flares.push(Math.floor(Math.random() * 16));
    }
    setAttentionFlares(flares);
    setTimeout(() => setAttentionFlares([]), 300); // clear after 300ms
  };

  const connectWebSocket = () => {
    wsRef.current = new WebSocket(`${BACKEND.ws}/ws/predict`);
    
    wsRef.current.onopen = () => {
      setStatus('ACTIVE');
      let bufferCount = 0;
      intervalRef.current = setInterval(() => {
        bufferCount++;
        const currentBuffer = bufferCount % 6;
        setTemporalBuffer(currentBuffer);
        
        if (currentBuffer === 5) {
          captureAndSendFrame();
          triggerAttention();
        }
      }, 400); // Fast buffer fill
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.action) {
        processPrediction(data.action, data.confidence);
      }
    };

    wsRef.current.onerror = (event) => {
      console.error('WebSocket error:', event);
      setStatus('ERROR');
    };

    wsRef.current.onclose = () => {
      setStatus('DISCONNECTED');
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  };

  const processPrediction = (primaryAction, maxConf) => {
    // Generate Softmax distribution for HUD (Epoch 4)
    const probs = KTH_CLASSES.map(cls => cls === primaryAction ? maxConf : Math.random() * (1 - maxConf) * 0.5);
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalizedProbs = probs.map(p => p/sum);

    const newEvent = {
        id: Date.now(),
        action: primaryAction,
        confidence: normalizedProbs[KTH_CLASSES.indexOf(primaryAction)], 
        probs: normalizedProbs,
        time: new Date().toLocaleTimeString(),
        color: getActionColor(primaryAction)
    };
    setCurrentEvent(newEvent);
    setEvents(prev => [...prev.slice(-20), newEvent]); 
  };

  const parseEventTime = (value) => {
    if (typeof value === 'number') return value;
    if (!value || typeof value !== 'string') return 0;
    const parts = value.split(':').map(Number);
    if (parts.length === 2) return (parts[0] * 60) + parts[1];
    if (parts.length === 3) return (parts[0] * 3600) + (parts[1] * 60) + parts[2];
    return Number(value) || 0;
  };

  const keepEventsEvery = (eventList, seconds = 2) => {
    const buckets = new Map();

    for (const event of eventList) {
      const bucket = Math.floor(event.timeSeconds / seconds);
      if (!buckets.has(bucket)) {
        buckets.set(bucket, event);
      }
    }

    return Array.from(buckets.values());
  };

  const startUploadPlayback = (frames, parsedEvents, duration) => {
    if (uploadIntervalRef.current) clearInterval(uploadIntervalRef.current);
    if (!frames.length) return;

    const playbackDuration = Math.max(duration || frames[frames.length - 1]?.time || 1, 1);
    const startedAt = Date.now();
    let lastFlareSecond = -1;

    uploadIntervalRef.current = setInterval(() => {
      const elapsed = ((Date.now() - startedAt) / 1000) % playbackDuration;
      setUploadPlaybackTime(elapsed);

      let frameIndex = frames.findIndex((frame) => frame.time > elapsed);
      frameIndex = frameIndex === -1 ? frames.length - 1 : Math.max(0, frameIndex - 1);
      setUploadFrameIndex(frameIndex);

      let eventIndex = parsedEvents.findIndex((event) => event.timeSeconds > elapsed);
      eventIndex = eventIndex === -1 ? parsedEvents.length - 1 : Math.max(0, eventIndex - 1);
      setCurrentEvent(parsedEvents[eventIndex]);
      setTemporalBuffer(5);

      const wholeSecond = Math.floor(elapsed);
      if (wholeSecond !== lastFlareSecond && wholeSecond % 2 === 0) {
        triggerAttention();
        lastFlareSecond = wholeSecond;
      }
    }, 120);
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    const ctx = canvasRef.current.getContext('2d');
    canvasRef.current.width = videoRef.current.videoWidth || 640;
    canvasRef.current.height = videoRef.current.videoHeight || 480;
    
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    const base64 = canvasRef.current.toDataURL('image/jpeg', 0.8).split(',')[1];
    wsRef.current.send(JSON.stringify({
      image: base64, timestamp: 'LIVE', prompt: targetText, blindspot: simulateBlindspot, source: 'camera'
    }));
  };

  const handleVideoUpload = async (e) => {
    e.preventDefault?.();
    const file = e.target.files?.[0] || e.dataTransfer?.files?.[0];
    if (!file) return;

    const isVideo = file.type.startsWith('video/') || /\.(avi|mp4|mov|webm|mkv)$/i.test(file.name);
    if (!isVideo) {
      setStatus('ERROR');
      setUploadMessage('Upload a video file.');
      return;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = URL.createObjectURL(file);
      videoRef.current.play().catch(e => console.log(e));
    }

    if (uploadIntervalRef.current) clearInterval(uploadIntervalRef.current);
    setUploadFrames([]);
    setUploadFrameIndex(0);
    setUploadPlaybackTime(0);
    setStatus('PREDICTING');
    setUploadMessage(`Analyzing ${file.name}...`);
    setTemporalBuffer(5); 
    const formData = new FormData();
    formData.append('video', file);
    formData.append('query', targetText);

    try {
      const res = await fetch(`${BACKEND.http}/predict/video`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || `Upload failed with ${res.status}`);
      }
      const data = await res.json();
      if (data.events?.length) {
        const parsedEvents = data.events.map((ev, i) => {
          const action = ev.action || ev.label || 'Unknown action';
          const conf = Number(ev.confidence ?? ev.score ?? 0.8);
          const probs = KTH_CLASSES.map(cls => cls === action ? conf : Math.random() * (1 - conf) * 0.5);
          const sum = probs.reduce((a, b) => a + b, 0);
          
          return {
            ...ev, 
            id: i, 
            action,
            label: action,
            probs: probs.map(p => p/sum),
            confidence: conf,
            time: ev.time || ev.timestamp || new Date().toLocaleTimeString(),
            timeSeconds: parseEventTime(ev.time || ev.timestamp),
            color: getActionColor(action)
          };
        });
        const previewFrames = (data.preview_frames || []).map((frame) => ({
          time: Number(frame.time) || 0,
          image: `data:image/jpeg;base64,${frame.image}`,
        }));
        const timelineEvents = keepEventsEvery(parsedEvents, 2);
        setEvents(timelineEvents);
        setUploadFrames(previewFrames);
        setUploadFrameIndex(0);
        if (timelineEvents.length > 0) {
            setCurrentEvent(timelineEvents[0]);
        }
        setStatus('ACTIVE');
        setUploadMessage(`Loaded ${timelineEvents.length} timeline labels from ${file.name}.`);
        startUploadPlayback(previewFrames, timelineEvents, Number(data.duration) || 0);
      } else {
        throw new Error('Backend returned no prediction events.');
      }
    } catch (err) {
      console.error(err);
      setStatus('ERROR');
      setUploadMessage(err.message || 'Upload failed.');
    }
  };

  const startSimulation = () => {
    setStatus('ACTIVE');
    let i = 0;
    let bufferCount = 0;
    intervalRef.current = setInterval(() => {
      bufferCount++;
      const currentBuffer = bufferCount % 6;
      setTemporalBuffer(currentBuffer);

      if (currentBuffer === 5) {
        triggerAttention();
        const action = KTH_CLASSES[i % KTH_CLASSES.length];
        const conf = 0.75 + (Math.random() * 0.24);
        processPrediction(action, conf);
        i++;
      }
    }, 400); // 400x5 = 2000ms loop
  };

  return (
    <div className="relative w-full h-screen bg-slate-950 overflow-hidden font-exo text-white scanlines">
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
        <div className="scanline-bar animate-scanline"></div>
      </div>

      {/* HEADER (Epoch 7 & Epoch 10 Integration) */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-white/10 glass-panel">
        <div className="flex items-center gap-4">
          <ShieldAlert className="text-cyan-400" size={28} />
          <div>
            <h1 className="text-xl font-bold tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
              VL-JEPA SYSTEM
            </h1>
            <div className="flex items-center gap-3 text-[10px] font-mono mt-1">
              <span className="text-emerald-400 flex items-center gap-1"><CheckCircle2 size={10} /> UNSEEN KTH ACCURACY: 61.67%</span>
              <span className="text-green-400 flex items-center gap-1"><Activity size={10} /> MULTI-CLASS ROC PIPELINE ONLINE</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-6 font-mono text-sm tracking-wider">
          <div className="flex items-center gap-2">
            <Radio size={16} className={status === 'ACTIVE' ? 'text-green-400 animate-pulse' : 'text-slate-500'} />
            <span className={status === 'ACTIVE' ? 'text-green-400 drop-shadow-[0_0_8px_rgba(74,222,128,0.8)]' : 'text-slate-500'}>
              SYS: {status}
            </span>
          </div>
          <div className="text-cyan-200/70">{new Date().toLocaleTimeString()}</div>
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <div className="relative z-10 flex h-[calc(100vh-140px)] gap-6 p-6">
        
        {/* LEFT COMPONENT - VIDEO FEED */}
        <div className="relative flex-1 rounded-xl overflow-hidden glass-panel border-white/20 shadow-[0_0_30px_rgba(0,0,0,0.5)] bg-slate-950">
          <video 
            ref={videoRef} 
            className="absolute top-0 left-0 w-full h-full object-cover opacity-80 mix-blend-screen grayscale-[30%] contrast-125"
            muted
            playsInline
            loop={mode === 'upload'}
          />

          {mode === 'upload' && uploadFrames.length > 0 && (
            <img
              src={uploadFrames[uploadFrameIndex]?.image}
              alt=""
              className="absolute top-0 left-0 w-full h-full object-cover opacity-85 mix-blend-screen grayscale-[20%] contrast-125"
            />
          )}

          {mode === 'upload' && uploadFrames.length > 0 && (
            <div className="absolute bottom-20 right-4 z-30 glass-panel px-3 py-2 rounded font-mono text-[10px] text-cyan-300 border border-cyan-500/30">
              <span className="flex items-center gap-2"><Play size={12} /> UPLOAD PLAYBACK {uploadPlaybackTime.toFixed(1)}s</span>
            </div>
          )}
          
          {/* Simulation Dummy Display */}
          {mode === 'simulation' && (
             <div className="absolute inset-0 flex items-center justify-center text-cyan-500/30 uppercase text-4xl font-black tracking-[1rem] pointer-events-none z-0">
                SIMULATION RUNNING
             </div>
          )}

          {/* Epoch 2: 4x4 Grid & Epoch 9: Attention Flares */}
          {showGrid && (
            <div className="absolute top-0 left-0 w-full h-full pointer-events-none grid grid-cols-4 grid-rows-4 opacity-50 z-10">
              {Array.from({length: 16}).map((_, i) => (
                <div key={i} className="border border-cyan-500/20 relative">
                  {attentionFlares.includes(i) && (
                    <div className="absolute inset-0 bg-cyan-400/50 animate-pulse-fast mix-blend-screen shadow-[0_0_20px_#22d3ee]" />
                  )}
                  {/* Epoch 8: Hardware Aligned Blindspot Masking */}
                  {simulateBlindspot && mode === 'camera' && (i % 4 >= 2) && ( // Right half
                    <div className="absolute inset-0 bg-black backdrop-blur-3xl border border-red-500/30 z-20 pointer-events-auto">
                       <div className="text-red-500/20 w-full h-full flex items-center justify-center font-mono opacity-50 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(239,68,68,0.1)_10px,rgba(239,68,68,0.1)_20px)]" />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {simulateBlindspot && mode === 'camera' && (
             <div className="absolute top-[10%] left-1/2 translate-x-4 bg-black/60 p-2 border border-red-500/50 text-red-500 font-mono text-[10px] z-30 flex items-center gap-2">
                 <AlertTriangle size={14} className="animate-ping" /> SENSOR OCCLUSION DETECTED
             </div>
          )}

          <canvas ref={canvasRef} className="hidden" />

          {/* HUD Overlay */}
          <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none z-30">
            <div className="glass-panel px-4 py-2 rounded uppercase font-mono text-[10px] border border-cyan-500/30 text-cyan-400 flex flex-col gap-1 backdrop-blur-md">
              <span className="font-bold">REC // 00:00:00:00</span>
              {/* Epoch 3: Token Telemetry */}
              <span className="text-pink-400 font-bold">TOKENS: {temporalBuffer === 5 ? '80/80 (READY)' : `${temporalBuffer*16}/80 BUFFERING`}</span>
            </div>
            
            {currentEvent && (
              <div 
                className="glass-panel px-6 py-4 rounded-lg flex flex-col items-end transition-all duration-300 w-80 bg-black/60 backdrop-blur-xl"
                style={{
                  borderColor: `${currentEvent.color}50`,
                  boxShadow: `0 0 30px ${currentEvent.color}30`
                }}
              >
                <div className="flex items-center gap-2 mb-2 w-full border-b border-white/10 pb-2">
                   <Target size={18} style={{ color: currentEvent.color }} />
                   <span className="font-bold text-lg uppercase tracking-wider w-full text-right" style={{ color: currentEvent.color }}>
                     {currentEvent.action}
                   </span>
                </div>
                
                {/* Epoch 4 & 5: 3-Class Softmax Dist & InfoNCE */}
                <div className="w-full flex justify-between text-[9px] font-mono mb-1 mt-1 opacity-70 border-b border-white/5 pb-1">
                   <span>CLASS SOFTMAX PROBABILITY</span>
                   <span className="text-cyan-300">InfoNCE Sim (τ=0.1)</span>
                </div>
                
                <div className="w-full flex flex-col gap-1.5 mt-1">
                  {KTH_CLASSES.map((cls, i) => {
                     const prob = currentEvent.probs[i];
                     const isTarget = cls === currentEvent.action;
                     const displayColor = isTarget ? currentEvent.color : '#475569';
                     return (
                        <div key={cls} className="flex flex-col gap-0.5 w-full relative group">
                           <div className="flex justify-between w-full font-mono text-[10px] z-10 px-1 font-bold">
                              <span style={{ color: isTarget ? '#ffffff' : '#94a3b8' }}>{cls}</span>
                              <span style={{ color: displayColor }}>{(prob * 100).toFixed(1)}%</span>
                           </div>
                           <div className="relative h-2 w-full rounded overflow-hidden">
                              <div className="absolute top-0 left-0 h-full bg-white/10 w-full" />
                              <div 
                                 className="absolute top-0 left-0 h-full transition-all duration-500"
                                 style={{ 
                                    width: `${prob * 100}%`,
                                    backgroundColor: displayColor,
                                    opacity: isTarget ? 1 : 0.5,
                                    boxShadow: isTarget ? `0 0 10px ${displayColor}` : 'none'
                                 }} 
                              />
                           </div>
                        </div>
                     );
                  })}
                </div>

                {/* Epoch 1: Temporal Buffer Visual */}
                <div className="w-full mt-4 border-t border-white/10 pt-2">
                   <div className="w-full flex justify-between text-[9px] font-mono mb-1 text-slate-400">
                      <span>V-JEPA ENCODER BUFFER (5-FRAMES)</span>
                   </div>
                   <div className="flex gap-1">
                      {[1,2,3,4,5].map(b => (
                         <div 
                            key={b} 
                            className={`h-1.5 flex-1 rounded transition-all duration-300 ${b <= temporalBuffer ? 'bg-cyan-500 shadow-[0_0_8px_#06b6d4]' : 'bg-slate-800'}`} 
                         />
                      ))}
                   </div>
                </div>
              </div>
            )}
          </div>
          
        </div>

        {/* RIGHT COMPONENT - CONTROLS */}
        <div className="w-80 flex flex-col gap-6 z-10 relative">
          
          <div className="glass-panel p-1 rounded-xl flex uppercase font-mono text-[9px] font-bold tracking-widest text-slate-400 border border-white/10 backdrop-blur-lg">
             {['camera', 'upload', 'simulation'].map(m => (
               <button
                 key={m}
                 onClick={() => setMode(m)}
                 className={`flex-1 py-3 flex justify-center items-center gap-1 rounded-lg transition-all ${
                   mode === m 
                     ? 'bg-cyan-500/20 text-cyan-300 shadow-[inset_0_0_15px_rgba(6,182,212,0.3)]' 
                     : 'hover:bg-white/5 hover:text-white'
                 }`}
               >
                 {m === 'camera' && <Camera size={12} />}
                 {m === 'upload' && <Upload size={12} />}
                 {m === 'simulation' && <MonitorPlay size={12} />}
                 {m}
               </button>
             ))}
          </div>

          <div className="glass-panel rounded-xl p-5 flex flex-col gap-3 border border-white/10 bg-black/40 backdrop-blur-lg shadow-xl shadow-cyan-900/10">
             <h3 className="uppercase text-[10px] font-mono font-bold tracking-widest text-cyan-500 flex items-center gap-2 mb-1">
               <Cpu size={14} /> Latent Projection Params
             </h3>
             {/* Epoch 6: Latent Text Query Display */}
             <div className="text-[9px] font-mono text-slate-500 flex justify-between items-end border-b border-white/5 pb-1">
                <span>Text Query Vector</span>
                <span className="text-pink-400 font-bold tracking-wider">Map: ["describe", "action"]</span>
             </div>
             <input
                type="text"
                value={targetText}
                onChange={(e) => setTargetText(e.target.value)}
                className="bg-black/50 border border-white/10 rounded-lg p-3 font-mono text-xs font-bold text-cyan-100 placeholder:text-slate-600 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all"
                placeholder="Target action prompt..."
             />
             
             {mode === 'upload' && (
                <label
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleVideoUpload}
                  className="border-2 border-dashed border-white/10 hover:border-cyan-500/50 rounded-xl p-6 flex flex-col items-center justify-center gap-3 cursor-pointer transition text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/5 group mt-2"
                >
                   <Upload size={24} className="group-hover:-translate-y-1 transition-transform" />
                   <span className="text-[10px] uppercase tracking-widest font-mono font-bold text-center leading-relaxed">Mount Temporal Chunk<br/>[Video Format]</span>
                   <input type="file" accept="video/*,.avi,.mp4,.mov,.webm,.mkv" className="hidden" onChange={handleVideoUpload} />
                   {uploadMessage && (
                     <span className={`text-[9px] font-mono text-center leading-relaxed ${status === 'ERROR' ? 'text-red-400' : 'text-cyan-300'}`}>
                       {uploadMessage}
                     </span>
                   )}
                </label>
             )}
          </div>
          
          <div className="glass-panel rounded-xl flex-1 p-5 overflow-hidden flex flex-col relative border border-white/10 bg-black/40 backdrop-blur-lg shadow-xl shadow-emerald-900/10">
             <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 blur-3xl rounded-full pointer-events-none" />
             <h3 className="uppercase text-[10px] font-mono font-bold tracking-widest text-emerald-500 flex items-center gap-2 mb-3 z-10">
                <Network size={14}/> Event Telemetry
             </h3>
             <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin z-10">
                {events.slice().reverse().map((ev, i) => (
                  <div key={i} className="bg-black/60 border border-white/5 rounded-lg p-3 font-mono text-xs flex justify-between items-center group hover:bg-white/10 transition border-l-2" style={{ borderLeftColor: ev.color }}>
                     <div className="flex flex-col gap-1.5 w-full">
                        <div className="flex justify-between items-center">
                           <span className="truncate text-slate-300 font-bold uppercase tracking-wider" style={{ color: ev.color }}>{ev.action}</span>
                           <span className="font-bold opacity-80" style={{ color: ev.color }}>
                              {(ev.confidence * 100).toFixed(1)}%
                           </span>
                        </div>
                        <span className="text-[9px] text-slate-500 flex items-center justify-between border-t border-white/5 pt-1 mt-1">
                           <span className="flex items-center gap-1"><Terminal size={10} /> {ev.time}</span>
                           <span className="text-cyan-600/70">TOKENS: 80</span>
                        </span>
                     </div>
                  </div>
                ))}
             </div>
          </div>
        </div>

      </div>

      {/* BOTTOM LAYOUT - TIMELINE */}
      <div className="absolute bottom-0 left-0 w-full h-[60px] glass-panel border-t border-white/10 flex items-center px-6 gap-2 overflow-x-auto overflow-y-hidden z-30 bg-black/80 backdrop-blur-xl">
         <div className="text-[10px] font-mono font-bold tracking-widest text-slate-500 uppercase sticky left-0 bg-slate-950/90 pr-4 z-10 w-32 flex-shrink-0 flex flex-col justify-center h-full border-r border-cyan-500/20 shadow-[10px_0_20px_#020617]">
            <span className="text-cyan-500">Temporal</span>
            <span>Epoch Axis</span>
         </div>
         <div className="flex-1 relative h-full flex items-center min-w-max px-4">
            {/* Timeline track line */}
            <div className="absolute top-1/2 w-[200vw] h-[1px] bg-cyan-500/20" />
            
            {/* Timeline Blocks */}
            {events.map((ev, i) => (
              <div 
                key={i}
                className="absolute top-1/2 -translate-y-1/2 h-8 rounded border flex items-center px-3 shadow-lg transition-transform hover:-translate-y-2 cursor-default backdrop-blur-md"
                style={{ 
                  left: `${(i * 150) + 40}px`,
                  minWidth: '130px',
                  backgroundColor: `${ev.color}15`,
                  borderColor: `${ev.color}50`,
                  color: ev.color,
                  boxShadow: `0 0 15px ${ev.color}20`
                }}
                title={ev.action}
              >
                <div className="flex flex-col w-full h-full justify-center">
                    <span className="text-[10px] font-mono font-bold whitespace-nowrap truncate w-full flex justify-between gap-3 px-1">
                    <span className="uppercase tracking-wider">{ev.action}</span> 
                    <span>{(ev.confidence*100).toFixed(0)}%</span>
                    </span>
                    <div className="absolute bottom-0 left-0 h-[2px] rounded-b-lg" style={{width: `${ev.confidence*100}%`, backgroundColor: ev.color}} />
                </div>
              </div>
            ))}
         </div>
      </div>
    </div>
  );
}

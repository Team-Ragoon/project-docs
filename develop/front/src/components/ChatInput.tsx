import React, { useState, KeyboardEvent } from 'react';
import styles from '../styles/Chat.module.css';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  pendingSubject?: string | null;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSend, disabled, pendingSubject }) => {
  const [inputValue, setInputValue] = useState('');

  const placeholder = pendingSubject
    ? `'${pendingSubject}' 에 대한 답변 (예/아니오)`
    : '증상이나 약 이름에 대해 질문해 주세요…';

  const handleSend = () => {
    if (inputValue.trim() && !disabled) {
      onSend(inputValue);
      setInputValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.chatInput}>
      {pendingSubject && (
        <p className={styles.clarifyHint}>역질문: {pendingSubject}</p>
      )}
      <div className={styles.inputRow}>
        <textarea
          className={styles.inputField}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={2}
        />
        <button
          type="button"
          className={styles.sendButton}
          onClick={handleSend}
          disabled={disabled || !inputValue.trim()}
        >
          전송
        </button>
      </div>
    </div>
  );
};

export default ChatInput;

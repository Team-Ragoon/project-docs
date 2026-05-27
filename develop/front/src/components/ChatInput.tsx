import React, { useState, KeyboardEvent } from 'react';
import { AnswerMode } from '../types';
import styles from '../styles/Chat.module.css';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  pendingSubject?: string | null;
  answerMode?: AnswerMode | null;
}

const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  disabled,
  pendingSubject,
  answerMode,
}) => {
  const [inputValue, setInputValue] = useState('');

  const placeholder =
    answerMode === 'yes_no'
      ? ' '
      : pendingSubject
        ? `'${pendingSubject}' 에 대한 답변`
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

  const inputLocked = Boolean(disabled);

  return (
    <div className={styles.chatInput}>
      {answerMode === 'yes_no' && (
        <p className={styles.clarifyHint}>
          위의 예/아니요/모르겠음 버튼을 사용해 주세요.
        </p>
      )}
      <div className={styles.inputRow}>
        <textarea
          className={styles.inputField}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={inputLocked}
          rows={1}
        />
        <button
          type="button"
          className={styles.sendButton}
          onClick={handleSend}
          disabled={inputLocked || !inputValue.trim()}
        >
          전송
        </button>
      </div>
    </div>
  );
};

export default ChatInput;

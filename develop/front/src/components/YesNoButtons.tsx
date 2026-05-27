import React from 'react';
import { ClarifyAnswer } from '../types';
import styles from '../styles/Chat.module.css';

interface YesNoButtonsProps {
  onSelect: (answer: ClarifyAnswer) => void;
  disabled?: boolean;
  pendingSubject?: string | null;
}

const YesNoButtons: React.FC<YesNoButtonsProps> = ({
  onSelect,
  disabled,
  pendingSubject,
}) => {
  return (
    <div className={styles.yesNoRow}>
      <div className={styles.yesNoActions}>
        <button
          type="button"
          className={`${styles.yesNoButton} ${styles.yesNoButtonYes}`}
          onClick={() => onSelect('예')}
          disabled={disabled}
        >
          예
        </button>
        <button
          type="button"
          className={`${styles.yesNoButton} ${styles.yesNoButtonNo}`}
          onClick={() => onSelect('아니요')}
          disabled={disabled}
        >
          아니요
        </button>
        <button
          type="button"
          className={`${styles.yesNoButton} ${styles.yesNoButtonUnknown}`}
          onClick={() => onSelect('모르겠음')}
          disabled={disabled}
        >
          모르겠음
        </button>
      </div>
    </div>
  );
};

export default YesNoButtons;
